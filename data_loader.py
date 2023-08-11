from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from util import log
import numpy as np
import torch
from PIL import Image


def make_training_and_validation_sets(dataset_path, validation_split=0.2):
    """
    Constructs and returns training and validation subsets from a root dataset
    path.
    """
    assert 0 <= validation_split <= 1

    dataset = ImageFolder(
        root=dataset_path,
        transform=make_common_image_transforms(),
    )
    training_set, validation_set = random_split(
        dataset, [1 - validation_split, validation_split]
    )
    log.info(f"Validation split: {validation_split}")
    log.info(f"Training dataset: {len(training_set)} samples")
    log.info(f"Validation dataset: {len(validation_set)} samples")
    return training_set, validation_set


def make_training_and_validation_loaders(
    dataset_path, batch_size, validation_split=0.2, balanced=False
):
    training_set, validation_set = make_training_and_validation_sets(
        dataset_path, validation_split
    )
    if balanced:
        training_loader = make_balanced_training_loader(training_set, batch_size)
        validation_loader = make_balanced_training_loader(validation_set, batch_size)
    else:
        training_loader = make_training_loader(training_set, batch_size)
        validation_loader = make_training_loader(validation_set, batch_size)
    return training_loader, validation_loader


def make_balanced_training_loader(data_set, batch_size):
    """
    Creates and returns a dataloader with a balanced sampler for subsets of a
    torch Dataset object. This should be used if the dataset is significantly
    unbalanced.
    """
    indices = data_set.indices
    class_labels = [data_set.dataset.targets[i] for i in indices]
    class_sample_count = np.bincount(class_labels)
    class_weights = 1.0 / class_sample_count
    sample_weights = np.array([class_weights[t] for t in class_labels])
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights), len(sample_weights)
    )
    return DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )


def make_training_loader(data_set, batch_size):
    """
    Creates and returns a standard dataloader for use in model training. This
    can be used if the dataset is sufficiently balanced.
    """
    return DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
    )


def make_evaluation_loader(testset_path):
    """
    Creates and returns a dataloader from a testset path for for use in model
    evaluation.
    """
    testset = ImageFolder(root=testset_path, transform=make_common_image_transforms())
    log.info(f"Testset: {len(testset)} samples detected")
    log.info(f"Testset: detected classes: {testset.class_to_idx}")
    return DataLoader(testset, shuffle=True)


def make_common_image_transforms(resize_shape=(256, 256), grayscale=True):
    """
    CommonImageTransforms provides common transformations used for ensuring
    that input images conform to dimensions expected by the model.
    """
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(num_output_channels=3))
    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Resize(resize_shape, antialias=True),
        ]
    )
    return transforms.Compose(transform_list)


def process_single_image(image_path: str, resize_shape=(256, 256), grayscale=True):
    image = Image.open(image_path)
    transform = make_common_image_transforms(
        resize_shape=resize_shape, grayscale=grayscale
    )

    # normalize and add batch dimension
    return transform(image).unsqueeze(0)
