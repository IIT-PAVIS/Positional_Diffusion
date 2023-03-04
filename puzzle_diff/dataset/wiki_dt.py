import json
import random
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

SEED = 42


class Wiki_dt(Dataset):
    def __init__(self, split="train"):
        super().__init__()
        data_path = Path(f"datasets/wiki/wiki_movie_plots_deduped.csv")
        split_len = 0.8 if split == "train" else ".2"
        rows = load_data(data_path)
        random.Random(SEED).shuffle(rows)
        self.examples = rows[: int(len(self.examples) * split_len)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


### Reordering task
def load_data(in_file):
    """
    Loads the dataset file:
    in_file: json file
    Returns a list of tuples (input, output)
    """

    df = pd.read_csv(in_file)
    rows = [[l for l in x.split(".")[:20] if len(l)] for x in df["Plot"]]

    return rows


if __name__ == "__main__":
    dt = Wiki_dt()
    print(dt[0])
