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


from dataset.dataset_utils import get_dataset


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

    _, test_dt, puzzle_sizes = get_dataset(dataset=dataset, puzzle_sizes=puzzle_sizes)

    dl_test = torch_geometric.loader.DataLoader(
        test_dt, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )

    # model = GNN_Diffusion.load_from_checkpoint("epoch=539-step=135000.ckpt")
    model = GNN_Diffusion.load_from_checkpoint("epoch=659-step=165000.ckpt")
    model.initialize_torchmetrics(puzzle_sizes)

    ### define training

    franklin = True if gpus > 1 else False

    experiment_name = f"{dataset}-{puzzle_sizes}"

    tags = [f"{dataset}", f'{"franklin" if franklin else "fisso"}', "train"]

    wandb_logger = WandbLogger(
        project="Puzzle-Diff",
        settings=wandb.Settings(code_dir="."),
        offline=True,
        name=experiment_name,
        tags=tags,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=gpus,
        strategy="ddp" if gpus > 1 else None,
        logger=wandb_logger,
        callbacks=[ModelSummary(max_depth=2)],
    )
    trainer.test(model, dl_test)


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