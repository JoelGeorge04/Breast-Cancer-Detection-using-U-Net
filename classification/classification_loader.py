import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

TARGET_SIZE = 128  # Image size for training

class BreastCancerDataset(Dataset):
    """
    PyTorch Dataset for breast cancer classification
    Folder structure:
        root/
            0/  (non-cancerous)
                image1.png
                image2.png
                ...
            1/  (cancerous)
                image1.png
                image2.png
                ...
    """
    def __init__(self, root, transform=None, target_size=TARGET_SIZE):
        self.root = root
        self.transform = transform
        self.target_size = target_size
        self.images = []
        self.labels = []
        
        # Load all image paths and labels
        for class_label in ["0", "1"]:
            class_dir = os.path.join(root, class_label)
            
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
            
            files = os.listdir(class_dir)
            print(f"Found {len(files)} images in class {class_label}")
            
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, file)
                    self.images.append(img_path)
                    self.labels.append(int(class_label))
        
        print(f"Total images loaded: {len(self.images)}")
        print(f"Class 0 (non-cancerous): {self.labels.count(0)}")
        print(f"Class 1 (cancerous): {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            # Return a blank image if loading fails
            img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        
        # Resize
        img = cv2.resize(img, (self.target_size, self.target_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float and normalize to [0, 1]
        img = img.astype("float32") / 255.0
        
        # Apply transforms if any
        if self.transform:
            img = self.transform(img)
        else:
            # Convert to tensor (H, W, C) -> (C, H, W)
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)
        
        label = torch.tensor(label, dtype=torch.float32)
        
        return img, label


def get_data_transforms(augment=True):
    """
    Get data augmentation transforms
    """
    if augment:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
    else:
        train_transform = None
    
    return train_transform


def load_classification_data(root, batch_size=8, shuffle=True, augment=True, num_workers=0, balance=False):
    """
    Create DataLoader for classification.

    Args:
        balance: If True, use WeightedRandomSampler to equalise class frequencies
                 per-epoch (useful when class counts differ).
    """
    from torch.utils.data import WeightedRandomSampler

    transform = get_data_transforms(augment) if augment else None
    dataset = BreastCancerDataset(root, transform=transform)

    sampler = None
    if balance and len(dataset) > 0:
        n0 = dataset.labels.count(0)
        n1 = dataset.labels.count(1)
        # Weight each sample inversely proportional to its class frequency
        weight_per_class = {0: 1.0 / max(n0, 1), 1: 1.0 / max(n1, 1)}
        sample_weights = [weight_per_class[lbl] for lbl in dataset.labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # sampler and shuffle are mutually exclusive

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # Prevent batch norm error with incomplete batches
    )

    return dataloader, dataset


def load_simple_classification_data(root, target_size=TARGET_SIZE):
    """
    Simple loader that returns all data as tensors (original style)
    """
    images = []
    labels = []
    
    print(f"Loading data from: {root}")
    
    for class_label in ["0", "1"]:
        class_dir = os.path.join(root, class_label)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue
        
        files = os.listdir(class_dir)
        print(f"Loading {len(files)} images from class {class_label}...")
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(class_dir, file)
                
                # Load image
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                # Resize
                img = cv2.resize(img, (target_size, target_size))
                
                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Normalize to [0, 1]
                img = img.astype("float32") / 255.0
                
                # Convert to tensor format (C, H, W)
                img = np.transpose(img, (2, 0, 1))
                
                images.append(img)
                labels.append(float(class_label))
    
    if len(images) == 0:
        print("No images found!")
        return None, None
    
    # Convert to tensors
    images = torch.from_numpy(np.array(images))
    labels = torch.from_numpy(np.array(labels))
    
    print(f"Loaded {len(images)} images total")
    print(f"Class 0: {(labels == 0).sum().item()}")
    print(f"Class 1: {(labels == 1).sum().item()}")
    
    return images, labels


if __name__ == "__main__":
    # Test the dataset
    print("Testing BreastCancerDataset...")
    dataset = BreastCancerDataset("final_dataset/images", transform=get_data_transforms(True))
    
    if len(dataset) > 0:
        img, label = dataset[0]
        print(f"\nSample image shape: {img.shape}")
        print(f"Sample label: {label}")
        
        # Test DataLoader
        print("\nTesting DataLoader...")
        dataloader, _ = load_classification_data("final_dataset/images", batch_size=4)
        
        for batch_imgs, batch_labels in dataloader:
            print(f"Batch images shape: {batch_imgs.shape}")
            print(f"Batch labels shape: {batch_labels.shape}")
            print(f"Labels: {batch_labels}")
            break
