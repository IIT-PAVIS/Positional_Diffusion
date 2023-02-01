from .puzzle_dataset import Puzzle_Dataset, Puzzle_Dataset_MP
from torchvision.datasets import CelebA
from torchvision.datasets import CIFAR100
from .wikiart_dt import Wikiart_DT

ALLOWED_DT = ["celeba", "cifar100", "wikiart"]


def get_dataset(dataset: str, puzzle_sizes: list) -> tuple:
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
        train_dt = CelebA(root="./datasets", download=True, split="train")
        test_dt = CelebA(root="./datasets", download=True, split="test")
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
    )
    puzzleDt_test = Puzzle_Dataset(
        dataset=test_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
    )

    return (puzzleDt_train, puzzleDt_test, real_puzzle_sizes)


def get_dataset_missing_pieces(dataset: str, puzzle_sizes: list) -> tuple:
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
        train_dt = CelebA(root="./datasets", download=True, split="train")
        test_dt = CelebA(root="./datasets", download=True, split="test")
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
        missing_perc=10,
    )
    puzzleDt_test = Puzzle_Dataset_MP(
        dataset=test_dt,
        dataset_get_fn=get_fn,
        patch_per_dim=real_puzzle_sizes,
        missing_perc=10,
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
