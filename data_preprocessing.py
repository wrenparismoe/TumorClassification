import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_data_loaders(dataset_dir="./data", batch_size=16, train_split=0.7, val_split=0.15):
    """
    Prepare data loaders for training, validation, and testing.
    
    Args:
        dataset_dir (str): Path to the dataset directory.
        batch_size (int): Batch size for the data loaders.
        train_split (float): Proportion of data for training (default: 70%).
        val_split (float): Proportion of data for validation (default: 15%).

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """

    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # (256, 256) or (avg_width, avg_height)
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to range [-1, 1]
    ])

    # Load the dataset from the specified directory
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

    # Define dataset splits. Default: 70% train, 15% val, 15% test
    train_size = int(train_split * len(full_dataset))
    val_size = int(val_split * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Get dataset class mapping
    class_names = full_dataset.classes  # Class labels

    return train_loader, val_loader, test_loader, class_names

