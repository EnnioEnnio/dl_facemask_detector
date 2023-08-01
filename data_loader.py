from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from util import log
import numpy as np
import torch


def make_training_and_validation_sets(dataset, validation_split=0.2):
    training_set, validation_set = random_split(
        dataset, [1 - validation_split, validation_split]
    )
    log.info(f"Validation split: {validation_split}")
    log.info(f"Training dataset: {len(training_set)} samples")
    log.info(f"Validation dataset: {len(validation_set)} samples")
    return training_set, validation_set


def make_balanced_dataloader(set, batch_size):
    # creates a balanced sampler for subsets of a torch Dataset object
    indices = set.indices
    class_labels = [set.dataset.targets[i] for i in indices]
    class_sample_count = np.bincount(class_labels)
    class_weights = 1.0 / class_sample_count
    sample_weights = np.array([class_weights[t] for t in class_labels])
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights), len(sample_weights)
    )
    return DataLoader(
        set,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        sampler=sampler,
    )


def make_training_loader():
    pass


def make_evaluation_loader(testset_path):
    testset = ImageFolder(
        root=testset_path,
        transform=common_transforms,
    )
    log.info(f"Testset: {len(testset)} samples detected")
    log.info(f"Testset: detected classes: {testset.class_to_idx}")
    return DataLoader(testset, shuffle=True)


common_transforms = transforms.Compose(
    [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Resize((256, 256), antialias=True),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)
