import argparse
import os
import sys
from typing import Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import math

import einops
import pytorch_lightning as pl
import torch
from lightning_modules.Diffusion_network_2 import DiffusionModel_Cond
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100, CelebA, Flickr8k
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Resize,
    ToPILImage,
    ToTensor,
)

import wandb


def main(batch_size, gpus, steps):
    wandb_logger = WandbLogger(
        project="Puzzle-Diff", settings=wandb.Settings(code_dir=".")
    )

    IMAGE_SIZE = 64
    PATCHES = 4  # Patches per dim
    patch_size = IMAGE_SIZE // PATCHES
    xx = einops.repeat(torch.linspace(-1, 1, PATCHES), "b -> b k1", k1=patch_size)
    y, x = torch.meshgrid(xx.flatten(), xx.flatten())
    y = torch.flip(y, (0,))
    pos_emb = torch.stack((x, y), 0)

    transform_tr = Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
            transforms.Lambda(lambda t: addPosEnc(t, pos_emb)),
        ]
    )

    transform_val = Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
            transforms.Lambda(lambda t: addPosEnc(t, pos_emb)),
            transforms.Lambda(lambda t: permute(t, PATCHES)),
        ]
    )

    def permute(
        t: torch.Tensor, patches_per_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_generate a permutation of the image
        and the relative gt pos embedding

        Args:
            t (torch.Tensor): tensor containg the image and the positional embedding [c, w, h]
            patches_per_dim (int): number of patches per dim

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the permutated image and the permutation
        """

        img_patches = einops.rearrange(
            t,
            " c (p1 w) (p2 h) -> c (p1 p2) w h",
            p1=patches_per_dim,
            p2=patches_per_dim,
        )  # split the image in patches, concatenate the patches in the first dimension [c, patches, w, h]

        permutation = torch.randperm(patches_per_dim**2)

        permutated_img_patches = img_patches[:, permutation]
        permutated_img = einops.rearrange(
            permutated_img_patches,
            "c (p1 p2) w h -> c (p1 w) (p2 h)",
            p1=patches_per_dim,
            p2=patches_per_dim,
        )

        return (permutated_img, permutation)

    def addPosEnc(t, pos):
        # patch_size = t.shape[1] // patches
        # ones = torch.ones((1, patch_size, patch_size))
        # tl = torch.cat((ones * -1, ones))
        # tr = torch.cat((ones, ones))
        # bl = torch.cat((ones * -1, ones * -1))
        # br = torch.cat((ones, ones * -1))
        # top = torch.cat((tl, tr), 2)
        # bot = torch.cat((bl, br), 2)
        # pos = torch.cat((top, bot), 1)
        t = torch.cat((t, pos), 0)

        return t

    channels = 2
    # batch_size = 16

    # dataset_tr = CelebA(
    #     root="./dataset", download=True, split="train", transform=transform_tr
    # )
    dataset_tr = CIFAR100(
        root=".dataset", train=True, transform=transform_tr, download=True
    )
    dataloader_tr = DataLoader(
        dataset_tr, batch_size=batch_size, shuffle=True, num_workers=8
    )

    # dataset_val = CelebA(
    #     root="./dataset", download=True, split="test", transform=transform_val
    # )

    dataset_val = CIFAR100(
        root=".dataset", train=False, transform=transform_val, download=True
    )
    dataloader_val = DataLoader(dataset_val, batch_size=4, shuffle=False, num_workers=8)

    model = DiffusionModel_Cond(
        steps=steps,
        channels=channels,
        patch_size=patch_size,
        patches=PATCHES,
        prefix="LightningAttCond",
        save_and_sample_every=math.floor(len(dataloader_tr) / gpus / 4),
        sampling="DDPM",
    )
    # model = DiffusionModel_Cond.load_from_checkpoint("Model_4x4_cond.pth")

    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="loss",
        mode="min",
        filename="{epoch:02d}-{global_step}",
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy="ddp",
        limit_val_batches=10,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
        # fast_dev_run=True,
        # overfit_batches=0.01,
    )

    trainer.fit(model, dataloader_tr, dataloader_val)
    trainer.test(model, dataloader_val)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=12)
    ap.add_argument("-gpus", type=int, default=1)
    ap.add_argument("-steps", type=int, default=200)
    args = ap.parse_args()
    main(batch_size=args.batch_size, gpus=args.gpus, steps=args.steps)
