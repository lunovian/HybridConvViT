import torchvision.transforms as transforms

def get_augmentations():
    """
    Returns a set of transformations to augment the training dataset.
    """
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    return train_transforms, val_transforms