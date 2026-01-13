from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=128, data_dir='./data', val_split=0.1, num_workers=2):
    """
    Load CIFAR-10 dataset, apply transforms, return DataLoaders (train, val, test).
    
    Args:
        batch_size (int): Batch size for training/eval
        data_dir (str): Directory to store data
        val_split (float): Fraction of training data to use for validation
        num_workers (int): Number of worker threads for loading data
    
    Returns:
        train_loader, val_loader, test_loader, num_classes, img_size (channels, height, width)
    """
    # Mean and Std for CIFAR-10 (for RGB)
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    num_classes = 10
    img_size = (3, 32, 32)
    
    full_train_set = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    val_size = int(len(full_train_set) * val_split)
    train_size = len(full_train_set) - val_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    # note: same transform to val set as train set here to simplify the code
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Loaded CIFAR-10: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test images.")
    
    return train_loader, val_loader, test_loader, num_classes, img_size
