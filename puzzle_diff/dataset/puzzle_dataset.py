import math
import random
from typing import List, Tuple

# import albumentations
# import cv2
import einops
import numpy as np
import torch
import torch_geometric as pyg
import torch_geometric.data as pyg_data
import torch_geometric.loader
import torchvision.transforms as transforms
from PIL import Image
from PIL.Image import Resampling
from torch import Tensor
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class RandomCropAndResizedToOriginal(transforms.RandomResizedCrop):
    def forward(self, img):
        size = img.size
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        return F.resized_crop(img, i, j, h, w, size, self.interpolation)


def _get_augmentation(augmentation_type: str = "none"):
    switch = {
        "weak": [transforms.RandomHorizontalFlip(p=0.5)],
        "hard": [
            transforms.RandomHorizontalFlip(p=0.5),
            RandomCropAndResizedToOriginal(
                size=(1, 1), scale=(0.8, 1), interpolation=InterpolationMode.BICUBIC
            ),
        ],
    }
    return switch.get(augmentation_type, [])


@torch.jit.script
def divide_images_into_patches(
    img, patch_per_dim: List[int], patch_size
) -> List[Tensor]:
    # img2 = einops.rearrange(img, "c h w -> h w c")

    # divide images in non-overlapping patches based on patch size
    # output dim -> a
    img2 = img.permute(1, 2, 0)
    patches = img2.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
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
        augment="",
    ) -> None:
        super().__init__()

        assert dataset is not None and dataset_get_fn is not None
        self.dataset = dataset
        self.dataset_get_fn = dataset_get_fn
        self.patch_per_dim = patch_per_dim
        self.augment = augment

        self.transforms = transforms.Compose(
            [
                *_get_augmentation(augment),
                transforms.ToTensor(),
            ]
        )

        self.patch_size = patch_size
        # self.tot_patches = patch_per_dim[0] * patch_per_dim[1]

    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)
        else:
            raise Exception("Dataset not provided")

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size
        img = img.resize((width, height), resample=Resampling.BICUBIC)
        img = self.transforms(img)

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
        augment=False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            dataset_get_fn=dataset_get_fn,
            patch_per_dim=patch_per_dim,
            patch_size=patch_size,
            augment=augment,
        )
        self.missing_pieces_perc = missing_perc

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size

        img = img.resize((width, height), resample=Resampling.BICUBIC)

        img = self.transforms(img)
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


class Puzzle_Dataset_ROT(Puzzle_Dataset):
    def __init__(
        self,
        dataset=None,
        dataset_get_fn=None,
        patch_per_dim=[(7, 6)],
        patch_size=32,
        augment=False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            dataset_get_fn=dataset_get_fn,
            patch_per_dim=patch_per_dim,
            patch_size=patch_size,
            augment=augment,
        )

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])

        rdim = torch.randint(len(self.patch_per_dim), size=(1,)).item()
        patch_per_dim = self.patch_per_dim[rdim]

        height = patch_per_dim[0] * self.patch_size
        width = patch_per_dim[1] * self.patch_size

        img = img.resize((width, height), resample=Resampling.BICUBIC)

        img = self.transforms(img)
        xy, patches = divide_images_into_patches(img, patch_per_dim, self.patch_size)

        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")

        patches_num = patches.shape[0]

        patches_numpy = (
            (patches * 255).long().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
        )
        patches_im = [Image.fromarray(patches_numpy[x]) for x in range(patches_num)]
        random_rot = torch.randint(low=0, high=4, size=(patches_num,))
        random_rot_one_hot = torch.nn.functional.one_hot(random_rot, 4)

        adj_mat = torch.ones(xy.shape[0], xy.shape[0])
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)

        # rotation classes : 0 -> no rotation
        #                   1 -> 90 degrees
        #                   2 -> 180 degrees
        #                   3 -> 270 degrees

        rots = torch.tensor(
            [
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],
            ]
        )

        rots_tensor = random_rot_one_hot @ rots
        rotated_patch = [
            x.rotate(rot * 90) for (x, rot) in zip(patches_im, random_rot)
        ]  # in PIL

        rotated_patch_tensor = [
            torch.tensor(np.array(patch)).permute(2, 0, 1).float() / 255
            for patch in rotated_patch
        ]

        patches = torch.stack(rotated_patch_tensor)
        xy = torch.cat([xy, rots_tensor], 1)

        data = pyg_data.Data(
            x=xy,
            patches=patches,
            edge_index=edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([patch_per_dim]),
        )
        return data


if __name__ == "__main__":
    from celeba_dt import CelebA_HQ

    train_dt = CelebA_HQ(train=True)
    dt = Puzzle_Dataset_ROT(
        train_dt, dataset_get_fn=lambda x: x[0], patch_per_dim=[(4, 4)]
    )

    dl = torch_geometric.loader.DataLoader(dt, batch_size=100)
    dl_iter = iter(dl)

    for i in range(5):
        k = next(dl_iter)
    pass
