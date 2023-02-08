import math
import random
from typing import List

import albumentations
import cv2
import einops
import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torch_geometric.loader
import torchvision.transforms as transforms
from PIL import Image
from torch import Tensor


@torch.jit.script
def divide_images_into_patches(
    img, patch_per_dim: List[int], patch_size
) -> List[Tensor]:
    # img2 = einops.rearrange(img, "c h w -> h w c")

    # divide images in non-overlapping patches based on patch size
    # output dim -> a
    # img2 = img.permute(1, 2, 0)
    patches = img.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    y = torch.linspace(-1, 1, patch_per_dim[0])
    x = torch.linspace(-1, 1, patch_per_dim[1])
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), -1)
    # print(patch_per_dim)

    return xy, patches


class Puzzle_Dataset(pyg_data.Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        train=False,
    ) -> None:
        super().__init__()

        assert dataset is not None and dataset_get_fn is not None
        self.dataset = dataset
        self.dataset_get_fn = dataset_get_fn
        self.patch_per_dim = patch_per_dim
        self.train = train
        if train:
            self.transforms = transforms.Compose([transforms.ToTensor()])
            self.Image_Aug = albumentations.Compose(
                [
                    albumentations.augmentations.HorizontalFlip(p=0.5),
                    albumentations.augmentations.ShiftScaleRotate(
                        shift_limit=0.05,
                        scale_limit=0.05,
                        rotate_limit=10,
                        interpolation=cv2.INTER_CUBIC,
                        p=1,
                    ),
                ]
            )

        else:
            self.transforms = transforms.Compose([transforms.ToTensor()])
        self.patch_size = patch_size
        # self.tot_patches = patch_per_dim[0] * patch_per_dim[1]

    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size
        img = img.resize((width, height))

        img = np.array(img)
        if self.train:
            img_aug = self.Image_Aug(image=img)
            img = img_aug["image"]
        img = torch.tensor(img / 255).float()

        # img = self.transforms(img)
        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        adj_mat = torch.ones(
            patch_per_dim[0] * patch_per_dim[1], patch_per_dim[0] * patch_per_dim[1]
        )
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)
        data = pyg_data.Data(
            x=xy,
            patches=patches,
            edge_index=edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


class Puzzle_Dataset_MP(Puzzle_Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        missing_perc=10,
        train=False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            dataset_get_fn=dataset_get_fn,
            patch_per_dim=patch_per_dim,
            patch_size=patch_size,
            train=train,
        )
        self.missing_pieces_perc = missing_perc

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size
        img = img.resize((width, height))

        # img = self.transforms(img)
        img = np.array(img)
        if self.train:
            img_aug = self.Image_Aug(image=img)
            img = img_aug["image"]
        img = torch.tensor(img / 255).float()
        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        num_pieces = xy.shape[0]
        pieces_to_remove = math.ceil(num_pieces * self.missing_pieces_perc / 100)

        perm = list(range(num_pieces))

        random.shuffle(perm)
        perm = perm[: num_pieces - pieces_to_remove]
        xy = xy[perm]
        patches = patches[perm]

        adj_mat = torch.ones(xy.shape[0], xy.shape[0])
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)
        data = pyg_data.Data(
            x=xy,
            patches=patches,
            edge_index=edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


if __name__ == "__main__":
    import torchvision

    celeba_get_fn = lambda x: x[0]
    celeba = torchvision.datasets.CelebA(root="./datasets", split="test", download=True)
    pdt = Puzzle_Dataset(dataset=celeba, dataset_get_fn=celeba_get_fn)

    dl = torch_geometric.loader.DataLoader(pdt, batch_size=100)
    dl_iter = iter(dl)

    for i in range(5):
        k = next(dl_iter)
    pass
