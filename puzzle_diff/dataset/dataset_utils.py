from torchvision.datasets import CIFAR100

from .celeba_dt import CelebA_HQ
from .nips_dt import Nips_dt
from .puzzle_dataset import Puzzle_Dataset, Puzzle_Dataset_MP, Puzzle_Dataset_ROT
from .roc_dt import Roc_dt
from .sind_dt import Sind_dt
from .sind_vist_dt import Sind_Vist_dt
from .text_dataset import Text_dataset
from .vist_dataset import Vist_dataset
from .wiki_dt import Wiki_dt
from .wikiart_dt import Wikiart_DT

ALLOWED_DT = ["celeba", "cifar100", "wikiart"]
ALLOWED_TEXT = ["nips", "sind", "roc", "wiki"]


def get_dataset(dataset: str, puzzle_sizes: list, augment=False) -> tuple:
    """
    Get dataset of images based on specified dataset name and puzzle sizes.

    Parameters:
    - dataset (str): The name of the dataset to be used (e.g., "celeba", "cifar100", "wikiart").
    - puzzle_sizes (list): A list of puzzle sizes to be used.

    Returns:
    - Tuple of three elements:
        - puzzleDt_train (Puzzle_Dataset): The training dataset.
        - puzzleDt_test (Puzzle_Dataset): The testing dataset.
        - real_puzzle_sizes (list): List of tuples containing the actual puzzle sizes.
    """

    # Check if the dataset is supported
    assert (
        dataset in ALLOWED_DT
    ), f"dataset {dataset} not supported, need to be one of {ALLOWED_DT}"

    # Create a list of tuples containing the actual puzzle sizes
    real_puzzle_sizes = [(x, x) for x in puzzle_sizes]

    # Define a lambda function to get the first element of a tuple
    get_fn = lambda x: x[0]

    # Load the specified dataset
    if dataset == "celeba":
        train_dt = CelebA_HQ(train=True)
        test_dt = CelebA_HQ(train=False)
    elif dataset == "cifar100":
        train_dt = CIFAR100(root="./datasets", download=True, train=True)
        test_dt = CIFAR100(root="./datasets", download=True, train=False)
    elif dataset == "wikiart":
        train_dt = Wikiart_DT(train=True)
        test_dt = Wikiart_DT(train=False)

    # Create puzzle datasets using the loaded datasets
    puzzleDt_train = Puzzle_Dataset(
        dataset=train_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        augment=augment,
    )
    puzzleDt_test = Puzzle_Dataset(
        dataset=test_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        augment=False,
    )

    return (puzzleDt_train, puzzleDt_test, real_puzzle_sizes)


def get_dataset_missing_pieces(
    dataset: str, puzzle_sizes: list, missing_pieces_perc: int, augment: bool = False
) -> tuple:
    """
    Get dataset of images based on specified dataset name and puzzle sizes.

    Parameters:
    - dataset (str): The name of the dataset to be used (e.g., "celeba", "cifar100", "wikiart").
    - puzzle_sizes (list): A list of puzzle sizes to be used.

    Returns:
    - Tuple of three elements:
        - puzzleDt_train (Puzzle_Dataset): The training dataset.
        - puzzleDt_test (Puzzle_Dataset): The testing dataset.
        - real_puzzle_sizes (list): List of tuples containing the actual puzzle sizes.
    """

    # Check if the dataset is supported
    assert (
        dataset in ALLOWED_DT
    ), f"dataset {dataset} not supported, need to be one of {ALLOWED_DT}"

    # Create a list of tuples containing the actual puzzle sizes
    real_puzzle_sizes = [(x, x) for x in puzzle_sizes]

    # Define a lambda function to get the first element of a tuple
    get_fn = lambda x: x[0]

    # Load the specified dataset
    if dataset == "celeba":
        train_dt = CelebA_HQ(train=True)
        test_dt = CelebA_HQ(train=False)
    elif dataset == "cifar100":
        train_dt = CIFAR100(root="./datasets", download=True, train=True)
        test_dt = CIFAR100(root="./datasets", download=True, train=False)
    elif dataset == "wikiart":
        train_dt = Wikiart_DT(train=True)
        test_dt = Wikiart_DT(train=False)

    # Create puzzle datasets using the loaded datasets
    puzzleDt_train = Puzzle_Dataset_MP(
        dataset=train_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        missing_perc=missing_pieces_perc,
        augment=augment,
    )
    puzzleDt_test = Puzzle_Dataset_MP(
        dataset=test_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        missing_perc=missing_pieces_perc,
        augment=False,
    )

    return (puzzleDt_train, puzzleDt_test, real_puzzle_sizes)


def get_dataset_ROT(dataset: str, puzzle_sizes: list, augment=False) -> tuple:
    """
    Get dataset of images based on specified dataset name and puzzle sizes.

    Parameters:
    - dataset (str): The name of the dataset to be used (e.g., "celeba", "cifar100", "wikiart").
    - puzzle_sizes (list): A list of puzzle sizes to be used.

    Returns:
    - Tuple of three elements:
        - puzzleDt_train (Puzzle_Dataset): The training dataset.
        - puzzleDt_test (Puzzle_Dataset): The testing dataset.
        - real_puzzle_sizes (list): List of tuples containing the actual puzzle sizes.
    """

    # Check if the dataset is supported
    assert (
        dataset in ALLOWED_DT
    ), f"dataset {dataset} not supported, need to be one of {ALLOWED_DT}"

    # Create a list of tuples containing the actual puzzle sizes
    real_puzzle_sizes = [(x, x) for x in puzzle_sizes]

    # Define a lambda function to get the first element of a tuple
    get_fn = lambda x: x[0]

    # Load the specified dataset
    if dataset == "celeba":
        train_dt = CelebA_HQ(train=True)
        test_dt = CelebA_HQ(train=False)
    elif dataset == "cifar100":
        train_dt = CIFAR100(root="./datasets", download=True, train=True)
        test_dt = CIFAR100(root="./datasets", download=True, train=False)
    elif dataset == "wikiart":
        train_dt = Wikiart_DT(train=True)
        test_dt = Wikiart_DT(train=False)

    # Create puzzle datasets using the loaded datasets
    puzzleDt_train = Puzzle_Dataset_ROT(
        dataset=train_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        augment=augment,
    )
    puzzleDt_test = Puzzle_Dataset_ROT(
        dataset=test_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        augment=False,
    )

    return (puzzleDt_train, puzzleDt_test, real_puzzle_sizes)


def get_dataset_old(
    dataset: str, puzzle_sizes: list
) -> tuple[Puzzle_Dataset, Puzzle_Dataset, list]:
    assert (
        dataset in ALLOWED_DT
    ), f"dataset {dataset} not supported, need to be one of {ALLOWED_DT}"

    real_puzzle_sizes = [(x, x) for x in puzzle_sizes]

    get_fn = lambda x: x[0]

    if dataset == "celeba":
        train_dt = CelebA(
            root="./datasets",
            download=True,
            split="train",
        )

        test_dt = CelebA(
            root="./datasets",
            download=True,
            split="test",
        )

    if dataset == "cifar100":
        train_dt = CIFAR100(
            root="./datasets",
            download=True,
            train=True,
        )

        test_dt = CIFAR100(root="./datasets", download=True, train=False)

    if dataset == "wikiart":
        train_dt = Wikiart_DT(train=True)
        test_dt = Wikiart_DT(train=False)

    puzzleDt_train = Puzzle_Dataset(
        dataset=train_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
    )

    puzzleDt_test = Puzzle_Dataset(
        dataset=test_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
    )

    return (puzzleDt_train, puzzleDt_test, real_puzzle_sizes)


def get_dataset_text(dataset: str):
    assert dataset in ALLOWED_TEXT

    if dataset == "nips":
        train_dt = Nips_dt(split="train")
        val_dt = Nips_dt(split="val")
        test_dt = Nips_dt(split="test")
    elif dataset == "sind":
        train_dt = Sind_dt(split="train")
        val_dt = Sind_dt(split="val")
        test_dt = Sind_dt(split="test")
    elif dataset == "roc":
        train_dt = Roc_dt(split="train")
        val_dt = Roc_dt(split="test")
        test_dt = Roc_dt(split="test")
    elif dataset == "wiki":
        train_dt = Wiki_dt(split="train")
        val_dt = Wiki_dt(split="test")
        test_dt = Wiki_dt(split="test")
    else:
        raise Exception(f"Dataset {dataset} is not provided.")

    train_dt = Text_dataset(train_dt)
    val_dt = Text_dataset(val_dt)
    test_dt = Text_dataset(test_dt)

    return train_dt, val_dt, test_dt


def get_dataset_vist(dataset: str):
    if dataset == "sind":
        train_dt = Sind_Vist_dt(split="train")
        test_dt = Sind_Vist_dt(split="test")
    else:
        raise Exception(f"Dataset {dataset} is not provided.")

    train_dt = Vist_dataset(train_dt)
    test_dt = Text_dataset(test_dt)

    return train_dt, None, test_dt
