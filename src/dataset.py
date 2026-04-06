import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# We expect our dataset to be organized in the following structure:
#   dataset/
#     train/  real/  fake/
#     valid/  real/  fake/
#     test/   real/  fake/
# PyTorch's ImageFolder will automatically assign labels based on folder names,
# so we need to make sure the folder names match our class names exactly.


# We define our main configuration constants here so they are easy to modify.
# IMAGE_SIZE is set to 224 because MobileNetV3 was originally trained on this resolution.
IMAGE_SIZE  = 224
# We use a batch size of 32 as a good balance between speed and memory usage.
# If we run out of RAM, we can reduce this to 16.
BATCH_SIZE  = 32
# We set NUM_WORKERS to 0 to avoid multiprocessing issues on Windows.
# On Linux or Mac we can increase this to speed up data loading.
NUM_WORKERS = 0

# We use the same mean and standard deviation values that were used to train
# MobileNetV3 on ImageNet. This is important because the pretrained model
# expects inputs normalized in the same way it was trained.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_transforms(split):
    # For the training split, we apply light data augmentation to artificially
    # increase the diversity of our training set and reduce overfitting.
    # We use random horizontal flips, small rotations, and slight changes
    # in brightness and contrast. These are common and safe augmentations
    # for news images that do not change the semantic meaning of the image.
    if split == "train":
        return transforms.Compose([
            # We resize all images to a fixed size so batches have consistent dimensions.
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            # We randomly flip images horizontally with a 50% probability.
            transforms.RandomHorizontalFlip(p=0.5),
            # We apply a small random rotation of up to 10 degrees in either direction.
            transforms.RandomRotation(degrees=10),
            # We randomly adjust brightness and contrast by up to 20%.
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            # We convert the PIL image to a PyTorch tensor with values in [0, 1].
            transforms.ToTensor(),
            # We normalize using ImageNet statistics to match the pretrained model's input.
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        # For validation and test splits, we only apply deterministic transformations.
        # We do not use any augmentation here because we want our evaluation
        # to be clean and reproducible across different runs.
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


def get_dataloaders(data_dir, batch_size=BATCH_SIZE):
    # We load all three splits at once and return them as a dictionary.
    # This makes it easy to access each split by name throughout the project.
    # The function returns three values:
    #   dataloaders   -> dict with keys "train", "valid", "test"
    #   dataset_sizes -> dict with the number of images in each split
    #   class_names   -> list of class names sorted alphabetically ["fake", "real"]

    data_dir = Path(data_dir)
    splits   = ["train", "valid", "test"]

    # We create an ImageFolder dataset for each split.
    # ImageFolder automatically finds all images in the subdirectories
    # and assigns integer labels based on the alphabetical order of folder names.
    # In our case: fake -> 0, real -> 1.
    datasets_dict = {
        split: datasets.ImageFolder(
            root      = data_dir / split,
            transform = get_transforms(split),
        )
        for split in splits
    }

    # We wrap each dataset in a DataLoader to handle batching and shuffling.
    # We only shuffle the training split to ensure randomness during training.
    # Shuffling validation and test would not affect results but could be confusing.
    dataloaders = {
        split: DataLoader(
            dataset     = datasets_dict[split],
            batch_size  = batch_size,
            shuffle     = (split == "train"),
            num_workers = NUM_WORKERS,
            # pin_memory speeds up the transfer of data to GPU if one is available.
            # It has no effect when running on CPU.
            pin_memory  = True,
        )
        for split in splits
    }

    # We store the number of images in each split so we can compute
    # epoch-level metrics like accuracy correctly during training.
    dataset_sizes = {split: len(datasets_dict[split]) for split in splits}

    # We extract the class names from the training dataset.
    # These are the folder names sorted alphabetically: ["fake", "real"].
    class_names = datasets_dict["train"].classes

    return dataloaders, dataset_sizes, class_names


if __name__ == "__main__":
    import sys

    # We allow the user to pass a custom dataset path as a command line argument.
    # If no argument is given, we default to the "dataset" folder in the project root.
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset"

    print(f"Loading dataset from: {data_dir}\n")
    loaders, sizes, classes = get_dataloaders(data_dir)

    # We print the class names and the number of images in each split
    # to quickly verify that the dataset was loaded correctly.
    print(f"Classes found: {classes}")
    for split, size in sizes.items():
        print(f"  {split:6s} -> {size:5d} images")

    # We load one batch from the training set to verify that
    # the shapes and labels are as expected.
    images, labels = next(iter(loaders["train"]))
    print(f"\nSample batch:")
    # We expect shape [batch_size, 3, 224, 224] for images
    # and [batch_size] for labels.
    print(f"  Image tensor shape : {images.shape}")
    print(f"  Label tensor shape : {labels.shape}")
    # We expect to see both 0 (fake) and 1 (real) in the labels.
    print(f"  Unique labels      : {labels.unique()}")
    print("\nDataset loaded successfully")