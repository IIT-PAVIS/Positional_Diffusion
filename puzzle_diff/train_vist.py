import argparse
import os
import sys

import torch_geometric

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

import argparse
import math
import random
import string

import pytorch_lightning as pl
from dataset.dataset_utils import get_dataset_vist
from model import spatial_diffusion_vist as sdv
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.loggers import WandbLogger

import wandb


def get_random_string(length):
    # choose from all lowercase letter
    letters = string.ascii_lowercase
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str  # print("Random string of length", length, "is:", result_str)


def main(
    batch_size,
    gpus,
    steps,
    num_workers,
    dataset,
    sampling,
    inference_ratio,
    offline,
    checkpoint_path,
    predict_xstart,
):
    ### Define dataset

    train_dt, _, test_dt = get_dataset_vist(dataset=dataset)

    dl_train = torch_geometric.loader.DataLoader(
        train_dt, batch_size=batch_size, num_workers=num_workers, shuffle=True
    )
    dl_test = torch_geometric.loader.DataLoader(
        test_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    ## DEFINE MODEL
    if sampling == "DDPM":
        inference_ratio = 1

    epoch_steps = len(dl_train) * 10
    max_steps = len(dl_train) * 100

    model = sdv.GNN_Diffusion(
        steps=steps,
        sampling=sampling,
        inference_ratio=inference_ratio,
        model_mean_type=sdv.ModelMeanType.EPISLON
        if not predict_xstart
        else sdv.ModelMeanType.START_X,
        warmup_steps=epoch_steps,
        max_train_steps=max_steps,
    )
    model.initialize_torchmetrics()

    ### define training

    franklin = True if gpus > 1 else False

    experiment_name = f"VIST-{dataset}-{steps}-{get_random_string(6)}"

    tags = [f"{dataset}", "text", "train"]

    wandb_logger = WandbLogger(
        project="Puzzle-Diff",
        settings=wandb.Settings(code_dir="."),
        offline=offline,
        name=experiment_name,
        # entity="puzzle_diff",
        entity="puzzle_diff_academic",
        tags=tags,
    )

    checkpoint_callback = ModelCheckpoint(monitor="sp", mode="max", save_top_k=2)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else None,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        num_sanity_val_steps=2,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
    )

    trainer.fit(model, dl_train, dl_test, ckpt_path=checkpoint_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=6)
    ap.add_argument("-gpus", type=int, default=1)
    ap.add_argument("-steps", type=int, default=100)
    ap.add_argument("-num_workers", type=int, default=8)
    ap.add_argument("-dataset", default="nips", choices=["sind"])
    ap.add_argument("-sampling", default="DDIM", choices=["DDPM", "DDIM"])
    ap.add_argument("-inference_ratio", type=int, default=10)
    ap.add_argument("--offline", action="store_true", default=False)
    ap.add_argument("--data_augmentation", type=str, default="none")
    ap.add_argument("--checkpoint_path", type=str, default="")
    ap.add_argument("--predict_xstart", type=bool, default=True)

    args = ap.parse_args()
    print(args)
    main(
        batch_size=args.batch_size,
        gpus=args.gpus,
        steps=args.steps,
        num_workers=args.num_workers,
        dataset=args.dataset,
        sampling=args.sampling,
        inference_ratio=args.inference_ratio,
        offline=args.offline,
        checkpoint_path=args.checkpoint_path,
        predict_xstart=args.predict_xstart,
    )
