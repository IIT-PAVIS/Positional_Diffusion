from typing import List
import torch_geometric as pyg
import torch_geometric.loader
import torch_geometric.data as pyg_data
from PIL import Image
import torchvision.transforms as transforms
import einops
import torch
from torch import Tensor


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

    return xy, patches


class Puzzle_Dataset(pyg_data.Dataset):
    def __init__(
        self, dataset=None, dataset_get_fn=None, patch_per_dim=(7, 6), patch_size=32
    ) -> None:

        super().__init__()

        assert dataset is not None and dataset_get_fn is not None
        self.dataset = dataset
        self.dataset_get_fn = dataset_get_fn
        self.patch_per_dim = patch_per_dim
        self.transforms = transforms.Compose([transforms.ToTensor()])
        self.height = patch_per_dim[0] * patch_size
        self.width = patch_per_dim[1] * patch_size
        self.patch_size = patch_size
        self.tot_patches = patch_per_dim[0] * patch_per_dim[1]

    def len(self) -> int:
        if self.dataset is not None:
            return len(self.dataset)

    def get(self, idx):
        if self.dataset is not None:
            img = self.dataset_get_fn(self.dataset[idx])
        img = img.resize((self.width, self.height))

        img = self.transforms(img)
        xy, patches = divide_images_into_patches(
            img, self.patch_per_dim, self.patch_size
        )
        adj_mat = torch.ones(self.tot_patches, self.tot_patches)
        edge_index, edge_attr = pyg.utils.dense_to_sparse(adj_mat)
        xy = einops.rearrange(xy, "x y c -> (x y) c")
        patches = einops.rearrange(patches, "x y c k1 k2 -> (x y) c k1 k2")
        data = pyg_data.Data(
            x=xy,
            patches=patches,
            edge_index=edge_index,
            ind_name=torch.tensor([idx]).long(),
            patches_dim=torch.tensor([self.patch_per_dim]),
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
