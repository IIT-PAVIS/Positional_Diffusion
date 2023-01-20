from pathlib import Path
from typing import Any
import pytorch_lightning as pl
import scipy
from functools import partial
from transformers.optimization import Adafactor
from .Transformer_GNN import Transformer_GNN

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

matplotlib.use("agg")


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


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class GNN_Diffusion(pl.LightningModule):
    def __init__(
        self,
        steps=600,
        sampling="DDPM",
        learning_rate=1e-4,
        save_and_sample_every=1000,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.learning_rate = learning_rate
        self.save_and_sample_every = save_and_sample_every

        ### DIFFUSION STUFF

        if sampling == "DDPM":
            self.p_sample = partial(self.p_sample, sampling_func=self.p_sample_ddpm)
            self.eta = 1
        elif sampling == "DDIM":
            self.p_sample = partial(self.p_sample, sampling_func=self.p_sample_ddim)
            self.eta = 0

        # define beta schedule
        self.betas = linear_beta_schedule(timesteps=steps)
        # self.betas = cosine_beta_schedule(timesteps=steps)
        # define alphas
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        ### BACKBONE

        self.visual_backbone = timm.create_model(
            "cspdarknet53", pretrained=True, features_only=True
        )

        self.steps = steps

        self.combined_features_dim = 4096 + 32 + 32

        # self.gnn_backbone = torch_geometric.nn.models.GAT(
        #     in_channels=self.combined_features_dim,
        #     hidden_channels=256,
        #     num_layers=2,
        #     out_channels=self.combined_features_dim,
        # )
        self.gnn_backbone = Transformer_GNN(
            self.combined_features_dim,
            hidden_dim=128 * 4,
            heads=4,
            output_size=self.combined_features_dim,
        )
        self.time_emb = nn.Embedding(self.steps, 32)
        self.pos_mlp = nn.Sequential(nn.Linear(2, 64), nn.GELU(), nn.Linear(64, 32))
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 128),
            nn.GELU(),
            nn.Linear(128, self.combined_features_dim),
        )
        self.final_mlp = nn.Sequential(
            nn.Linear(self.combined_features_dim, 64), nn.GELU(), nn.Linear(64, 2)
        )

        mean = torch.tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        std = torch.tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.save_hyperparameters()

    def forward(self, xy_pos, time, patch_rgb, edge_index) -> Any:
        # mean = patch_rgb.new_tensor([0.4850, 0.4560, 0.4060])[None, :, None, None]
        # std = patch_rgb.new_tensor([0.2290, 0.2240, 0.2250])[None, :, None, None]

        patch_rgb = (patch_rgb - self.mean) / self.std

        # fe[3].reshape(fe[0].shape[0],-1)
        patch_feats = self.visual_backbone.forward(patch_rgb)[3].reshape(
            patch_rgb.shape[0], -1
        )
        # patch_feats = patch_feats
        time_feats = self.time_emb(time)
        pos_feats = self.pos_mlp(xy_pos)
        combined_feats = torch.cat([patch_feats, pos_feats, time_feats], -1)
        combined_feats = self.mlp(combined_feats)
        feats = self.gnn_backbone(x=combined_feats, edge_index=edge_index)
        final_feats = self.final_mlp(feats + combined_feats)

        return final_feats

    # forward diffusion
    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses_old(self, x_start, t, noise=None, loss_type="l1", cond=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self(x_noisy, t, cond)

        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    def p_losses(
        self, x_start, t, noise=None, loss_type="l1", cond=None, edge_index=None
    ):

        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self(x_noisy, t, cond, edge_index)

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
    def p_sample_ddpm(self, x, t, t_index, cond, edge_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self(x, t, cond, edge_index) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_ddim(self, x, t, t_index, cond):
        if t[0] == 0:
            return x

        eta = self.eta
        alpha_prod = extract(self.alphas_cumprod, t, x.shape)

        if (t[0] - 1) == 0:
            alpha_prod_prev = alpha_prod * 0 + 1
        else:
            alpha_prod_prev = extract(self.alphas_cumprod, t - 1, x.shape)
        beta = 1 - alpha_prod
        beta_prev = 1 - alpha_prod_prev

        model_output = self(x, t, cond)

        # estimate x_0
        x_0 = (x - beta**0.5 * model_output) / alpha_prod**0.5
        variance = (beta_prev / beta) * (1 - alpha_prod / alpha_prod_prev)
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
    def p_sample_loop(self, shape, cond, edge_index):
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

        for i in tqdm(
            reversed(range(0, self.steps)),
            desc="sampling loop time step",
            total=self.steps,
        ):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                i,
                cond=cond,
                edge_index=edge_index,
            )
            imgs.append(img)
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
        return imgs

    @torch.no_grad()
    def p_sample(self, x, t, t_index, cond, edge_index, sampling_func):
        return sampling_func(x, t, t_index, cond, edge_index)

    @torch.no_grad()
    def sample(self, image_size, batch_size=16, channels=3, cond=None, edge_index=None):
        return self.p_sample_loop(
            shape=(batch_size, channels, image_size, image_size),
            cond=cond,
            edge_index=edge_index,
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
        )
        if batch_idx % self.save_and_sample_every == 0 and self.local_rank == 0:
            imgs = self.p_sample_loop(batch.x.shape, batch.patches, batch.edge_index)
            img = imgs[0]
            for i in range(4):
                fig, ax = plt.subplots(2, 2)
                idx = torch.where(batch.batch == i)[0]
                patches_rgb = batch.patches[idx]
                gt_pos = batch.x[idx]
                pos = img[idx]
                gt_img = self.create_image_from_patches(
                    patches_rgb, gt_pos, n_patches=batch.patches_dim[i], i=i
                )

                pred_img = self.create_image_from_patches(
                    patches_rgb, pos, n_patches=batch.patches_dim[i], i=i
                )
                ax[0, 0].imshow(gt_img)
                ax[0, 1].imshow(pred_img)
                ax[1, 0].scatter(gt_pos[:, 0].cpu(), gt_pos[:, 1].cpu())
                ax[1, 0].set_aspect("equal")

                ax[1, 1].scatter(pos[:, 0].cpu(), pos[:, 1].cpu())
                ax[1, 1].set_aspect("equal")
                ax[0, 0].set_title(f"{self.current_epoch}-{batch.ind_name[i]}")

                fig.canvas.draw()
                im = PIL.Image.frombytes(
                    "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
                )
                im = wandb.Image(im)
                self.logger.experiment.log(
                    {"image": im, "global_step": self.global_step + i}
                )

                plt.savefig(f"results/asd_{self.current_epoch}-{batch.ind_name[i]}.png")
                plt.close()
        self.log("loss", loss)

        return loss

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
        # plt.figure()
        # plt.imshow(new_image)
        # plt.savefig(f"asd_{i}.png")

    def training_step_old(self, batch, batch_idx):

        batch = batch[0]
        batch_size = batch.shape[0]

        results_folder = Path("./results")

        t = torch.randint(0, self.steps, (batch_size,), device=self.device).long()

        batch_2 = einops.rearrange(
            batch,
            "b c (w1 w) (h1 h) -> b (w1 h1) c w h",
            w1=self.patches,
            h1=self.patches,
        )

        batch_noise = batch_2[:, :, -2:]
        cond = batch_2[:, :, :3]

        loss = self.p_losses(batch_noise, t, loss_type="huber", cond=cond)

        if (
            batch_idx != 0
            and batch_idx % self.save_and_sample_every == 0
            and self.local_rank == 0
        ):

            milestone = batch_idx // self.save_and_sample_every
            batches = num_to_groups(4, batch_size)
            all_images_list = list(
                map(
                    lambda n: self.sample(
                        image_size=self.image_size,
                        batch_size=n,
                        channels=self.channels,
                        cond=cond[:n],
                    ),
                    batches,
                )
            )
            # all_images = torch.cat(all_images_list, dim=0)
            all_images = torch.tensor(np.stack(all_images_list[0], 0))
            all_images = (all_images + 1) * 0.5
            self.save_img(
                all_images[-1],
                str(
                    results_folder
                    / f"{self.logger.experiment.name}"
                    / f"{self.prefix}-{self.current_epoch}-{milestone}.png"
                ),
                self.logger.experiment,
                self.global_step,
            )
        self.log("loss", loss)

        return loss

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

    def on_test_epoch_start(self) -> None:
        if self.local_rank == 0:
            self.correct = 0
            self.num_images = 0
        # return super().on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        if self.local_rank == 0:

            logger = self.logger.experiment
            accuracy = self.correct / self.num_images
            print(f"Ep:{self.current_epoch:3d} - accuracy: {accuracy:0.4f}")
            logger.log({"accuracy": accuracy, "epoch": self.current_epoch})
        # return super().on_test_epoch_end()

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

    # Save or display the new image

    def test_step_old(
        self, batch, batch_idx, *args, **kwargs
    ):  # -> Optional[STEP_OUTPUT]:
        if self.local_rank != 0:
            return
        self.p_sample_loop(batch.x, batch.patches, batch.edge_index)

    def test_step_old(
        self, batch, batch_idx, *args, **kwargs
    ):  # -> Optional[STEP_OUTPUT]:
        if self.local_rank != 0:
            return

        permutations = batch[0][1]
        images = batch[0][0]

        conditioning = images[:, :3]
        pos = images[:, -2:]
        logger = self.logger.experiment

        # divide into patches
        cond_patches = img_to_patches(conditioning, self.patches)

        predicted_img_list = self.p_sample_loop(shape=pos.shape, cond=cond_patches)
        predicted_img = predicted_img_list[-1]
        # self.generate_video(predicted_img_list)

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
        batch_size = images.shape[0]

        matplotlib.use("agg")
        fig, ax = plt.subplots(3, 4)

        for i in range(batch_size):
            permutate_gt_pos = pos_patches[permutations[i]]

            img = (einops.rearrange(predicted_img[i][:3], "c w h -> w h c") + 1) * 0.5
            pos = einops.rearrange(
                predicted_img[i][-2:],
                "c (k1 w) (k2 h) -> (k1 k2) w h c",
                k1=self.patches,
                k2=self.patches,
            )
            pos = pos.mean((1, 2))

            cost = scipy.spatial.distance.cdist(permutate_gt_pos, pos, "euclidean")
            cost = np.log(1 + cost)
            ass = scipy.optimize.linear_sum_assignment(cost)
            correct = (ass[0] == ass[1]).all()
            self.num_images += 1
            if correct:
                self.correct += 1

            idx = scipy.optimize.linear_sum_assignment(cost)[1]
            img_patches = einops.rearrange(
                img,
                "(k1 w) (k2 h) c -> (k1 k2) w h c",
                k1=self.patches,
                k2=self.patches,
            )
            idx_sort = np.argsort(permutations[i].cpu())
            img_perm = img_patches[idx_sort]

            img_perm_original_shape = einops.rearrange(
                img_perm,
                "(k1 k2) w h c -> (k1 w) (k2 h) c",
                k1=self.patches,
                k2=self.patches,
            )
            new_img = einops.rearrange(
                img_perm[idx],
                "(k1 k2) w h c -> (k1 w) (k2 h) c",
                k1=self.patches,
                k2=self.patches,
            )
            ax[0][i].imshow(torch.tensor(img_perm_original_shape))
            for k in range(self.patches**2):
                ax[2][i].plot(pos[k][0], pos[k][1], "*", label=f"p-{k}")
                ax[2][i].set_xlim([-1, 2])
                ax[2][i].set_ylim([-1, 2])
                ax[2][i].set_aspect("equal")
            ax[1][i].imshow(torch.tensor(new_img))

        ax[2][-1].legend(bbox_to_anchor=(1, -0.18), loc="upper right", ncol=4)
        fig.subplots_adjust(bottom=0.3)
        ax[0][0].set_title(
            f"{self.prefix}-{self.current_epoch}-{self.global_step}-{batch_idx}"
        )

        fig.tight_layout()
        fig.canvas.draw()
        im = PIL.Image.frombytes(
            "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
        )
        im = wandb.Image(im)
        logger.log({"val_image": im, "global_step": self.global_step})

        results_folder = Path("./results")

        im_path = (
            results_folder
            / f"{self.logger.experiment.name}"
            / "val"
            / f"{self.prefix}-{self.current_epoch}-{self.global_step}-{batch_idx}.png"
        )

        im_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(f"{im_path}")

        pass
        # return super().test_step(*args, **kwargs)

    def on_validation_epoch_start(self) -> None:
        self.on_test_epoch_start()

    def on_validation_epoch_end(self) -> None:
        self.on_test_epoch_end()

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        return self.test_step(batch, batch_idx, *args, **kwargs)


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
