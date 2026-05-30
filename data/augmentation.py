"""
PyTorch transforms for data augmentation during training.

Separate train/val pipelines are provided. All augmentations are
conservative to match the narrow synthetic distribution.
"""

from torchvision import transforms

IMG_SIZE = 224


def get_train_transforms() -> transforms.Compose:
    """Augmentation pipeline used during training."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def get_val_transforms() -> transforms.Compose:
    """Deterministic pipeline used during validation / inference."""
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
