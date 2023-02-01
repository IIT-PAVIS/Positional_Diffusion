import argparse
import sys, os
from typing import Tuple

import torch_geometric

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from dataset.puzzle_dataset import Puzzle_Dataset
from dataset.wikiart_dt import Wikiart_DT  # , Wikiart_DT_pytables
from model.spatial_diffusion import GNN_Diffusion
import argparse
from torchvision.datasets import CelebA
import pytorch_lightning as pl
import math
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from pytorch_lightning.loggers import WandbLogger
import wandb


def main(batch_size, gpus, steps, num_workers):

    celeba_get_fn = lambda x: x[0]

    # celebA_tr = CelebA(
    #     root="./datasets",
    #     download=True,
    #     split="train",
    # )

    # celebA_test = CelebA(
    #     root="./datasets",
    #     download=True,
    #     split="test",
    # )
    dt_train = Wikiart_DT(train=True)
    dt_test = Wikiart_DT(train=False)

    puzzleDt_train = Puzzle_Dataset(
        dataset=dt_train,
        dataset_get_fn=celeba_get_fn,
        patch_per_dim=[
            (6, 6),
        ],
    )

    puzzleDt_test = Puzzle_Dataset(
        dataset=dt_test,
        dataset_get_fn=celeba_get_fn,
        patch_per_dim=[
            (6, 6),
        ],
    )

    dl_train = torch_geometric.loader.DataLoader(
        puzzleDt_train, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dl_test = torch_geometric.loader.DataLoader(
        puzzleDt_test, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )

    # dl_train = dl_test  # TODO <----------------- CHANGE to train once debugging

    save_and_sample_every = 20000  # math.floor(len(dl_train) / gpus / 4)

    model = GNN_Diffusion(
        steps=steps,
        sampling="DDIM",
        inference_ratio=10,
        save_and_sample_every=save_and_sample_every,
        # bb="DarkNet",
    )
    model.initialize_torchmetrics(
        [
            (6, 6),
        ]
    )

    wandb_logger = WandbLogger(
        project="Puzzle-Diff", settings=wandb.Settings(code_dir="."), offline=False
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="overall_acc", mode="max", save_top_k=2
    )

    from pytorch_lightning.profiler import AdvancedProfiler

    # prof = AdvancedProfiler(filename="prof.txt")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else None,
        # limit_val_batches=10,
        # limit_train_batches=10,
        # limit_val_batches=0.20,
        # max_epochs=1,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        # accumulate_grad_batches=10,
        # profiler=prof,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
    )
    trainer.fit(model, dl_train, dl_test)

    pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=4)
    ap.add_argument("-gpus", type=int, default=1)
    ap.add_argument("-steps", type=int, default=600)
    ap.add_argument("-num_workers", type=int, default=8)

    args = ap.parse_args()
    main(
        batch_size=args.batch_size,
        gpus=args.gpus,
        steps=args.steps,
        num_workers=args.num_workers,
    )
