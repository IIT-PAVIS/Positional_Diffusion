import argparse
import sys, os

import torch_geometric

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from model.spatial_diffusion import GNN_Diffusion
import argparse
import pytorch_lightning as pl
import math
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary

from pytorch_lightning.loggers import WandbLogger
import wandb


from dataset.dataset_utils import get_dataset_missing_pieces


def main(
    batch_size,
    gpus,
    steps,
    num_workers,
    dataset,
    puzzle_sizes,
    sampling,
    inference_ratio,
    offline,
):

    ### Define dataset

    train_dt, test_dt, puzzle_sizes = get_dataset_missing_pieces(
        dataset=dataset, puzzle_sizes=puzzle_sizes, missing_pieces_perc=30
    )

    dl_train = torch_geometric.loader.DataLoader(
        train_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    dl_test = torch_geometric.loader.DataLoader(
        test_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    ## DEFINE MODEL
    if sampling == "DDPM":
        inference_ratio = 1

    model = GNN_Diffusion(
        steps=steps,
        sampling=sampling,
        inference_ratio=inference_ratio,
    )
    model.initialize_torchmetrics(puzzle_sizes)

    ### define training

    franklin = True if gpus > 1 else False

    experiment_name = f"MissingPieces-{dataset}-{puzzle_sizes}"

    tags = [f"{dataset}", f'{"franklin" if franklin else "fisso"}', "train"]

    wandb_logger = WandbLogger(
        project="Puzzle-Diff",
        settings=wandb.Settings(code_dir="."),
        offline=offline,
        name=experiment_name,
        tags=tags,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="overall_acc", mode="max", save_top_k=2
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else None,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, ModelSummary(max_depth=2)],
    )
    trainer.fit(model, dl_train, dl_test)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument("-batch_size", type=int, default=32)
    ap.add_argument("-gpus", type=int, default=1)
    ap.add_argument("-steps", type=int, default=300)
    ap.add_argument("-num_workers", type=int, default=8)
    ap.add_argument(
        "-dataset", default="celeba", choices=["celeba", "wikiart", "cifar100"]
    )
    ap.add_argument("-sampling", default="DDIM", choices=["DDPM", "DDIM"])
    ap.add_argument("-inference_ratio", type=int, default=10)
    ap.add_argument(
        "-puzzle_sizes", nargs="+", default=[6], type=int, help="Input a list of values"
    )
    ap.add_argument("--offline", action="store_true", default=False)

    args = ap.parse_args()
    print(args)
    main(
        batch_size=args.batch_size,
        gpus=args.gpus,
        steps=args.steps,
        num_workers=args.num_workers,
        dataset=args.dataset,
        puzzle_sizes=args.puzzle_sizes,
        sampling=args.sampling,
        inference_ratio=args.inference_ratio,
        offline=args.offline,
    )
