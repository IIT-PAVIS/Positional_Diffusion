import argparse
import sys, os
from typing import Tuple

import torch_geometric

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from dataset.puzzle_dataset import Puzzle_Dataset
from model.spatial_diffusion import GNN_Diffusion
import argparse
from torchvision.datasets import CelebA
import pytorch_lightning as pl
import math
from pytorch_lightning.callbacks import ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger
import wandb


def main(batch_size, gpus, steps):

    celeba_get_fn = lambda x: x[0]

    celebA_tr = CelebA(
        root="./datasets",
        download=True,
        split="train",
    )

    celebA_test = CelebA(
        root="./datasets",
        download=True,
        split="test",
    )

    puzzleDt_train = Puzzle_Dataset(
        dataset=celebA_tr, dataset_get_fn=celeba_get_fn, patch_per_dim=(4, 4)
    )

    puzzleDt_test = Puzzle_Dataset(
        dataset=celebA_test, dataset_get_fn=celeba_get_fn, patch_per_dim=(4, 4)
    )

    dl_train = torch_geometric.loader.DataLoader(
        puzzleDt_train, batch_size=batch_size, num_workers=10, shuffle=False
    )
    dl_test = torch_geometric.loader.DataLoader(
        puzzleDt_test, batch_size=batch_size, num_workers=10, shuffle=False
    )

    # dl_train = dl_test  # TODO <----------------- CHANGE to train once debugging

    save_and_sample_every = math.floor(len(dl_train) / gpus / 4)

    model = GNN_Diffusion(
        steps=steps, sampling="DDPM", save_and_sample_every=save_and_sample_every
    )

    wandb_logger = WandbLogger(
        project="Puzzle-Diff", settings=wandb.Settings(code_dir="."), offline=False
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy="ddp",
        limit_val_batches=10,
        logger=wandb_logger,
        # callbacks=[checkpoint_callback]
    )
    trainer.fit(model, dl_train)

    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=32)
    ap.add_argument("-gpus", type=int, default=1)
    ap.add_argument("-steps", type=int, default=600)
    args = ap.parse_args()
    main(batch_size=args.batch_size, gpus=args.gpus, steps=args.steps)
