from pathlib import Path
from typing import Any
import pytorch_lightning as pl
import scipy
from functools import partial
from transformers.optimization import Adafactor

# from .backbones.Transformer_GNN import Transformer_GNN
from collections import defaultdict

# from .network_modules import (
#     default,
#     partial,
#     SinusoidalPositionEmbeddings,
#     PreNorm,
#     Downsample,
#     Upsample,
#     Residual,
#     LinearAttention,
#     ConvNextBlock,
#     ResnetBlock,
#     Attention,
#     exists,
# )
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from tqdm import tqdm
from torch.optim import Adam

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import wandb
from torch import Tensor
import torchvision
import timm
import torch_geometric.nn.models
from PIL import Image

import matplotlib
import torchmetrics
from .backbones import Dark_TFConv, Eff_GAT
import colorsys

matplotlib.use("agg")


def interpolate_color1d(color1, color2, fraction):
    # color1 = [float(x) / 255 for x in color1]
    # color2 = [float(x) / 255 for x in color2]
    hsv1 = color1  #  colorsys.rgb_to_hsv(*color1)
    hsv2 = color2  #  colorsys.rgb_to_hsv(*color2)
    h = hsv1[0] + (hsv2[0] - hsv1[0]) * fraction
    s = hsv1[1] + (hsv2[1] - hsv1[1]) * fraction
    v = hsv1[2] + (hsv2[2] - hsv1[2]) * fraction
    return tuple(x for x in (h, s, v))


def interpolate_color(
    pos, col_1=(1, 0, 0), col_2=(1, 1, 0), col_3=(0, 0, 1), col_4=(0, 1, 0)
):
    f1 = float((pos[0] + 1) / 2)
    f2 = float((pos[1] + 1) / 2)
    c1 = interpolate_color1d(col_1, col_2, f1)
    c2 = interpolate_color1d(col_3, col_4, f1)
    return interpolate_color1d(c1, c2, f2)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def cosine_beta_schedule(timesteps, s=0.08):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape=None):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out[:, None]  # out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


import torch


@torch.jit.script
def greedy_cost_assignment(pos1: torch.Tensor, pos2: torch.Tensor) -> torch.Tensor:
    # Compute pairwise distances between positions
    dist = torch.norm(pos1[:, None] - pos2, dim=2)

    # Create a tensor to store the assignments
    assignments = torch.zeros(dist.size(0), 3, dtype=torch.int64)

    # Create a mask to keep track of assigned positions
    mask = torch.ones_like(dist, dtype=torch.bool)

    # Counter for keeping track of the number of assignments
    counter = 0

    # While there are still unassigned positions
    while mask.sum() > 0:
        # Find the minimum distance
        min_val, min_idx = dist[mask].min(dim=0)

        # Get the indices of the two dimensions
        idx = int(min_idx.item())
        ret = mask.nonzero()[idx, :]
        i = ret[0]
        j = ret[1]

        # Add the assignment to the tensor
        assignments[counter, 0] = i
        assignments[counter, 1] = j
        assignments[counter, 2] = min_val

        # Increase the counter
        counter += 1

        # Remove the assigned positions from the distance matrix and the mask
        mask[i, :] = 0
        mask[:, j] = 0

    return assignments[:counter]


class GNN_Diffusion(pl.LightningModule):
    def __init__(
        self,
        steps=600,
        inference_ratio=1,
        sampling="DDPM",
        learning_rate=1e-4,
        save_and_sample_every=1000,
        bb=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.save_and_sample_every = save_and_sample_every

        ### DIFFUSION STUFF

        if sampling == "DDPM":
            self.inference_ratio = 1
            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddpm,
            )
            self.eta = 1
        elif sampling == "DDIM":
            self.inference_ratio = inference_ratio
            self.p_sample = partial(
                self.p_sample,
                sampling_func=self.p_sample_ddim,
            )
            self.eta = 0

        # define beta schedule
        betas = linear_beta_schedule(timesteps=steps)
        self.timesteps = torch.arange(0, 700).flip(0)
        self.register_buffer("betas", betas)
        # self.betas = cosine_beta_schedule(timesteps=steps)
        # define alphas
        alphas = 1.0 - self.betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.register_buffer("sqrt_recip_alphas", sqrt_recip_alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        self.steps = steps
        ### BACKBONE
        if bb == "DarkNet":
            self.model = Dark_TFConv(steps=steps)
        else:
            self.model = Eff_GAT(steps=steps)

        self.save_hyperparameters()

    def initialize_torchmetrics(self, n_patches):
        metrics = {}

        for i in n_patches:
            metrics[f"{i}_acc"] = torchmetrics.MeanMetric()
            metrics[f"{i}_nImages"] = torchmetrics.SumMetric()
        metrics["overall_acc"] = torchmetrics.MeanMetric()
        metrics["overall_nImages"] = torchmetrics.SumMetric()
        self.metrics = nn.ModuleDict(metrics)

    def forward(self, xy_pos, time, patch_rgb, edge_index, batch) -> Any:
        return self.model(xy_pos, time, patch_rgb, edge_index, batch)
        # # mean = patch_rgb.new_tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        # # std = patch_rgb.new_tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        # # if patch_feats == None:

        # patch_rgb = (patch_rgb - self.mean) / self.std

        # # fe[3].reshape(fe[0].shape[0],-1)
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        #     patch_rgb.shape[0], -1
        # )
        # # patch_feats = patch_feats
        # time_feats = self.time_emb(time)
        # pos_feats = self.pos_mlp(xy_pos)
        # combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        # combined_feats = self.mlp(combined_feats)
        # feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)
        # final_feats = self.final_mlp(feats + combined_feats)

        # return final_feats

    def forward_with_feats(
        self,
        xy_pos: Tensor,
        time: Tensor,
        patch_rgb: Tensor,
        edge_index: Tensor,
        patch_feats: Tensor,
        batch,
    ) -> Any:
        return self.model.forward_with_feats(
            xy_pos, time, patch_rgb, edge_index, patch_feats, batch
        )
        # mean = patch_rgb.new_tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        # std = patch_rgb.new_tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        # if patch_feats == None:

        # patch_rgb = (patch_rgb - self.mean) / self.std

        # # fe[3].reshape(fe[0].shape[0],-1)
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        # patch_rgb.shape[0], -1
        # )
        # patch_feats = patch_feats
        # time_feats = self.time_emb(time)
        # pos_feats = self.pos_mlp(xy_pos)
        # combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        # combined_feats = self.mlp(combined_feats)
        # feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)
        # final_feats = self.final_mlp(feats + combined_feats)

        # return final_feats

    def visual_features(self, patch_rgb):

        # patch_rgb = (patch_rgb - self.mean) / self.std

        # # fe[3].reshape(fe[0].shape[0],-1)
        # patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
        #     patch_rgb.shape[0], -1
        # )
        # return patch_feats
        return self.model.visual_features(patch_rgb)

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):

        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    # def p_losses_old(self, x_start, t, noise=None, loss_type="l1", cond=None):
    #     if noise is None:
    #         noise = torch.randn_like(x_start)

    #     x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
    #     predicted_noise = self(x_noisy, t, cond)

    #     if loss_type == "l1":
    #         loss = F.l1_loss(noise, predicted_noise)
    #     elif loss_type == "l2":
    #         loss = F.mse_loss(noise, predicted_noise)
    #     elif loss_type == "huber":
    #         loss = F.smooth_l1_loss(noise, predicted_noise)
    #     else:
    #         raise NotImplementedError()

    #     return loss

    def p_losses(
        self,
        x_start,
        t,
        noise=None,
        loss_type="l1",
        cond=None,
        edge_index=None,
        batch=None,
    ):

        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self(x_noisy, t, cond, edge_index, batch)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample_ddpm(self, x, t, t_index, cond, edge_index, patch_feats, batch):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x
            - betas_t
            * self.forward_with_feats(
                x, t, cond, edge_index, patch_feats=patch_feats, batch=batch
            )
            / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def _get_variance_old(self, timestep, prev_timestep):
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep]
            if prev_timestep >= 0
            else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    def _get_variance(self, timestep, prev_timestep):
        alpha_prod_t = extract(
            self.alphas_cumprod, timestep
        )  # self.alphas_cumprod[timestep]

        alpha_prod_t_prev = (
            extract(self.alphas_cumprod, prev_timestep)
            if (prev_timestep >= 0).all()
            else alpha_prod_t * 0 + 1
        )

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (
            1 - alpha_prod_t / alpha_prod_t_prev
        )

        return variance

    @torch.no_grad()
    def p_sample_ddim(
        self, x, t, t_index, cond, edge_index, patch_feats, batch
    ):  # (self, x, t, t_index, cond):
        # if t[0] == 0:
        #     return x

        prev_timestep = t - self.inference_ratio

        eta = self.eta
        alpha_prod = extract(self.alphas_cumprod, t, x.shape)

        if (prev_timestep >= 0).all():
            alpha_prod_prev = extract(self.alphas_cumprod, prev_timestep, x.shape)
        else:
            alpha_prod_prev = alpha_prod * 0 + 1

        beta = 1 - alpha_prod
        beta_prev = 1 - alpha_prod_prev

        model_output = self.forward_with_feats(
            x, t, cond, edge_index, patch_feats=patch_feats, batch=batch
        )

        # estimate x    _0
        x_0 = (x - beta**0.5 * model_output) / alpha_prod**0.5
        variance = self._get_variance(
            t, prev_timestep
        )  # (beta_prev / beta) * (1 - alpha_prod / alpha_prod_prev)
        std_eta = eta * variance**0.5

        # estimate "direction to x_t"
        pred_sample_direction = (1 - alpha_prod_prev - std_eta**2) ** (
            0.5
        ) * model_output

        prev_sample = alpha_prod_prev ** (0.5) * x_0 + pred_sample_direction

        if eta > 0:

            noise = torch.randn(model_output.shape, dtype=model_output.dtype).to(
                self.device
            )
            prev_sample = prev_sample + std_eta * noise
        return prev_sample

    # Algorithm 2 but save all images:
    @torch.no_grad()
    def p_sample_loop(self, shape, cond, edge_index, batch):
        # device = next(model.parameters()).device
        device = self.device

        b = shape[0]
        # start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        # img = einops.rearrange(
        #     img,
        #     "b c (w1 w) (h1 h) -> b (w1 h1) c w h",
        #     h1=self.patches,
        #     w1=self.patches,
        # )
        imgs = []

        patch_feats = self.visual_features(cond)

        # time_t = torch.full((b,), i, device=device, dtype=torch.long)

        time_t = torch.full((b,), 0, device=device, dtype=torch.long)

        for i in tqdm(
            list(reversed(range(0, self.steps, self.inference_ratio))),
            desc="sampling loop time step",
        ):
            img = self.p_sample(
                img,
                # torch.full((b,), i, device=device, dtype=torch.long),
                time_t + i,
                i,
                cond=cond,
                edge_index=edge_index,
                patch_feats=patch_feats,
                batch=batch,
            )
            # imgs.append(img)
            # if i is not None:  # == 0:
            # if i == 0:
            #     img2 = img.clone()

            #     img2 = torch.cat((cond, img2), 2)
            #     img2 = einops.rearrange(
            #         img2,
            #         "b (w1 h1) c w h -> b c (w1 w) (h1 h)",
            #         h1=self.patches,
            #         w1=self.patches,
            #     )
            #     imgs.append(img2.cpu().numpy())
        imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(
        self, x, t, t_index, cond, edge_index, sampling_func, patch_feats, batch
    ):
        return sampling_func(x, t, t_index, cond, edge_index, patch_feats, batch)

    @torch.no_grad()
    def sample(
        self,
        image_size,
        batch_size=16,
        channels=3,
        cond=None,
        edge_index=None,
        batch=None,
    ):
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size),
            cond=cond,
            edge_index=edge_index,
            batch=batch,
        )

    def configure_optimizers(self):
        # optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)
        # optimizer = Adafactor(self.parameters())
        optimizer = Adafactor(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):

        # return super().training_step(*args, **kwargs)
        batch_size = batch.batch.max().item() + 1
        t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()

        new_t = torch.gather(t, 0, batch.batch)

        loss = self.p_losses(
            batch.x,
            new_t,
            loss_type="huber",
            cond=batch.patches,
            edge_index=batch.edge_index,
            batch=batch.batch,
        )
        if batch_idx == 0 and self.local_rank == 0:
            imgs = self.p_sample_loop(
                batch.x.shape, batch.patches, batch.edge_index, batch=batch.batch
            )
            img = imgs[-1]

            save_path = Path(f"results/{self.logger.experiment.name}/train")
            for i in range(
                min(batch.batch.max().item(), 4)
            ):  # save max 4 images during training loop
                idx = torch.where(batch.batch == i)[0]
                patches_rgb = batch.patches[idx]
                gt_pos = batch.x[idx]
                pos = img[idx]
                n_patches = batch.patches_dim[i]
                i_name = batch.ind_name[i]
                self.save_image(
                    patches_rgb=patches_rgb,
                    pos=pos,
                    gt_pos=gt_pos,
                    patches_dim=n_patches,
                    ind_name=i_name,
                    file_name=save_path,
                )

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():

            imgs = self.p_sample_loop(
                batch.x.shape, batch.patches, batch.edge_index, batch=batch.batch
            )
            img = imgs[-1]

            if self.local_rank == 0 and batch_idx < 10:
                save_path = Path(f"results/{self.logger.experiment.name}/val")
                for i in range(
                    min(batch.batch.max().item(), 4)
                ):  # save max 4 images during training loop
                    idx = torch.where(batch.batch == i)[0]
                    patches_rgb = batch.patches[idx]
                    gt_pos = batch.x[idx]
                    pos = img[idx]
                    n_patches = batch.patches_dim[i]
                    i_name = batch.ind_name[i]
                    self.save_image(
                        patches_rgb=patches_rgb,
                        pos=pos,
                        gt_pos=gt_pos,
                        patches_dim=n_patches,
                        ind_name=i_name,
                        file_name=save_path,
                    )
            # accuracy_dict = defaultdict(lambda: [])
            for i in range(batch.batch.max() + 1):

                idx = torch.where(batch.batch == i)[0]
                patches_rgb = batch.patches[idx]
                gt_pos = batch.x[idx]
                pos = img[idx]
                n_patches = batch.patches_dim[i].tolist()
                i_name = batch.ind_name[i]

                y = torch.linspace(-1, 1, n_patches[0], device=self.device)
                x = torch.linspace(-1, 1, n_patches[1], device=self.device)
                xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
                real_grid = einops.rearrange(xy, "x y c-> (x y) c")

                gt_ass = greedy_cost_assignment(gt_pos, real_grid)
                sort_idx = torch.sort(gt_ass[:, 0])[1]
                gt_ass = gt_ass[sort_idx]

                pred_ass = greedy_cost_assignment(pos, real_grid)
                sort_idx = torch.sort(pred_ass[:, 0])[1]
                pred_ass = pred_ass[sort_idx]

                # assignement = greedy_cost_assignment(gt_pos, pos)

                # self.num_images+=1

                self.metrics[f"{tuple(n_patches)}_nImages"].update(1)
                self.metrics["overall_nImages"].update(1)
                if (gt_ass[:, 1] == pred_ass[:, 1]).all():
                    self.metrics[f"{tuple(n_patches)}_acc"].update(1)
                    self.metrics["overall_acc"].update(1)
                    # accuracy_dict[tuple(n_patches)].append(1)
                else:

                    self.metrics[f"{tuple(n_patches)}_acc"].update(0)
                    self.metrics["overall_acc"].update(0)
                    # accuracy_dict[tuple(n_patches)].append(0)

            self.log_dict(self.metrics)
        # return accuracy_dict

    def validation_epoch_end(self, outputs) -> None:
        self.log_dict(self.metrics)

    def test_epoch_end(self, outputs) -> None:
        return self.validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)
        # all_outs = self.all_gather(outputs)

        ## mean = torch.mean(all_outs)
        ## num_images = all_outs.shape[0]

        # if self.local_rank == 0:
        # out_dict = {}
        # for d in all_outs:
        # out_dict = {
        # k: out_dict.get(k, []) + d.get(k, [])
        # for k in out_dict.keys() | d.keys()
        # }

        # acc_dict = {}
        # overall_acc = []
        # for key, val in out_dict.items():
        ## acc_dict[key] = {}
        # arr = torch.stack(out_dict[key])
        # acc_dict[f"{key}_acc"] = arr.float().mean()
        # acc_dict[f"{key}_num_img"] = arr.shape[0]
        # overall_acc.append(arr.float().mean())
        # overall_acc = torch.stack(overall_acc).mean()  # torch.mean(overall_acc)
        # self.log(
        # "val",
        # {"epoch": self.current_epoch, "overall_acc": overall_acc, **acc_dict},
        # rank_zero_only=True,
        # )
        # self.log("val_acc", overall_acc, rank_zero_only=True)

    # def on_validation_epoch_start(self) -> None:
    #     self.accuracy_dict = defaultdict(lambda: [])

    # def validation_step(self, batch, batch_idx, *args, **kwargs):
    # return self.test_step(batch, batch_idx, *args, **kwargs)

    def create_image_from_patches(self, patches, pos, n_patches=(4, 4), i=0):

        patch_size = 32
        height = patch_size * n_patches[0]
        width = patch_size * n_patches[1]
        new_image = Image.new("RGB", (width, height))
        for p in range(patches.shape[0]):

            patch = patches[p, :]
            patch = Image.fromarray(
                ((patch.permute(1, 2, 0)) * 255).cpu().numpy().astype(np.uint8)
            )
            x = pos[p, 0] * (1 - 1 / n_patches[0])
            y = pos[p, 1] * (1 - 1 / n_patches[1])
            x_pos = int((x + 1) * width / 2) - patch_size // 2
            y_pos = int((y + 1) * height / 2) - patch_size // 2
            new_image.paste(patch, (x_pos, y_pos))
        return new_image

    def save_image(
        self, patches_rgb, pos, gt_pos, patches_dim, ind_name, file_name: Path
    ):
        file_name.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(2, 2)

        gt_img = self.create_image_from_patches(
            patches_rgb, gt_pos, n_patches=patches_dim, i=ind_name
        )
        # assignement = greedy_cost_assignment(pos, gt_pos)

        pred_img = self.create_image_from_patches(
            patches_rgb, pos, n_patches=patches_dim, i=ind_name
        )
        col = list(map(interpolate_color, gt_pos))
        ax[0, 0].imshow(gt_img)
        ax[0, 1].imshow(pred_img)
        ax[1, 0].scatter(gt_pos[:, 0].cpu(), gt_pos[:, 1].cpu(), c=col)
        ax[1, 0].invert_yaxis()
        ax[1, 0].set_aspect("equal")

        ax[1, 1].scatter(pos[:, 0].cpu(), pos[:, 1].cpu(), c=col)

        ax[1, 1].invert_yaxis()
        ax[1, 1].set_aspect("equal")
        ax[0, 0].set_title(f"{self.current_epoch}-{ind_name}")

        ax[0, 1].set_title(f"{patches_dim}")

        fig.canvas.draw()
        im = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        im = wandb.Image(im)
        self.logger.experiment.log(
            {f"{file_name.stem}": im, "global_step": self.global_step}
        )

        plt.savefig(f"{file_name}/asd_{self.current_epoch}-{ind_name}.png")
        plt.close()

    def save_img(self, img_list, file_str, logger, step):
        Path(file_str).parent.mkdir(exist_ok=True)
        xx = einops.repeat(
            torch.linspace(-1, 1, self.patches), "b -> b k1", k1=self.patch_size
        )
        y, x = torch.meshgrid(xx.flatten(), xx.flatten())
        y = torch.flip(y, (0,))
        pos_emb = torch.stack((x, y), 0)
        pos_patches = einops.rearrange(
            pos_emb,
            "c (k1 w) (k2 h) -> (k1 k2) w h c",
            k1=self.patches,
            k2=self.patches,
        ).mean((1, 2))
        pos_patches = (pos_patches + 1) * 0.5

        import scipy

        matplotlib.use("agg")
        fig, ax = plt.subplots(3, 4)
        for i in range(img_list.shape[0]):
            img = einops.rearrange(img_list[i][:3], "c w h -> w h c")
            pos = einops.rearrange(
                img_list[i][-2:],
                "c (k1 w) (k2 h) -> (k1 k2) w h c",
                k1=self.patches,
                k2=self.patches,
            )
            pos = pos.mean((1, 2))

            cost = scipy.spatial.distance.cdist(pos_patches, pos, "euclidean")

            idx = scipy.optimize.linear_sum_assignment(cost)[1]
            img_patches = einops.rearrange(
                img,
                "(k1 w) (k2 h) c -> (k1 k2) w h c",
                k1=self.patches,
                k2=self.patches,
            )
            new_img = einops.rearrange(
                img_patches[idx],
                "(k1 k2) w h c -> (k1 w) (k2 h) c",
                k1=self.patches,
                k2=self.patches,
            )
            ax[0][i].imshow(img)
            for k in range(self.patches**2):
                ax[2][i].plot(pos[k][0], pos[k][1], "*", label=f"p-{k}")
                ax[2][i].set_xlim([-1, 2])
                ax[2][i].set_ylim([-1, 2])
                ax[2][i].set_aspect("equal")
            ax[1][i].imshow(new_img)

        ax[2][-1].legend(bbox_to_anchor=(1, -0.18), loc="upper right", ncol=4)
        fig.subplots_adjust(bottom=0.3)

        fig.tight_layout()
        fig.canvas.draw()
        im = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        im = wandb.Image(im)
        logger.log({"image": im, "global_step": step})
        plt.savefig(f"{file_str}")
        plt.close()

    def generate_video(self, predicted_img_list):
        img1 = [im[0] for im in predicted_img_list]
        img1_stack = np.stack(img1, 0)
        img1_stack_pos = img1_stack[:, :, :, :]
        patches = einops.rearrange(
            img1_stack_pos,
            "im pos (k1 w) (k2 h) -> im pos (k1 k2) w h",
            k1=self.patches,
            k2=self.patches,
        )
        pos_patches = patches[:, -2:, :, :]
        pos_mean = pos_patches.mean((-2, -1))

        matplotlib.use("agg")

        # Create a blank image with desired size
        height = 64
        width = 64
        from PIL import Image

        patch_width = patch_height = 16

        # Iterate through the patches and their positions
        for i in tqdm(range(patches.shape[0])):

            new_image = Image.new("RGB", (64, 64))
            for p in range(patches.shape[2]):

                patch = patches[i, :3, p]
                patch = Image.fromarray(
                    ((patch.transpose(1, 2, 0) + 1) * 128).astype(np.uint8)
                )
                x = pos_mean[i, -2, p] * (1 - 1 / self.patches)
                y = -pos_mean[i, -1, p] * (1 - 1 / self.patches)
                x_pos = int((x + 1) * width / 2) - patch_width // 2
                y_pos = int((y + 1) * height / 2) - patch_height // 2
                new_image.paste(patch, (x_pos, y_pos))
            new_image.save(f"video/patches_{i:04d}.png")

        for i in tqdm(range(pos_mean.shape[0])):
            plt.figure()
            for j in range(pos_mean.shape[2]):
                plt.scatter(pos_mean[i, 0, j], pos_mean[i, 1, j])
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)
            plt.savefig(f"video/{i:04d}.png")
            plt.close()


def img_to_patches(t, patches_per_dim):
    return einops.rearrange(
        t,
        "b c (p1 w) (p2 h) -> b (p1 p2) c w h",
        p1=patches_per_dim,
        p2=patches_per_dim,
    )


def patches_to_img(t: Tensor, patches_per_dim):
    return einops.rearrange(
        t,
        "b (p1 p2) c w h-> b c (p1 w) (p2 h) ",
        p1=patches_per_dim,
        p2=patches_per_dim,
    )
