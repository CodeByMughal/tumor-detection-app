import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

IMG_SIZE = 224

# ── Transforms ────────────────────────────────────────────────────
# Training: augment images to increase dataset size artificially
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),       # flip randomly
    transforms.RandomRotation(15),           # rotate up to 15 degrees
    transforms.RandomAffine(0, shear=10),    # slight shear
    transforms.ColorJitter(brightness=0.3),  # vary brightness
    transforms.ToTensor(),                   # → tensor (0–1)
    transforms.Normalize([0.5], [0.5])       # center around 0
])

# Validation/Test: no augmentation — just clean preprocessing
val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


class MedicalScanDataset(Dataset):
    """
    Loads medical scan images from a folder structure:
    
    data/brain_mri/
        yes/  ← tumor images
        no/   ← normal images
    
    Works for any scan type — brain, lung, breast, AVN bone.
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image    = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def load_brain_mri_dataset(data_dir, batch_size=8, val_split=0.2):
    """
    Load brain MRI dataset from folder structure.
    Returns train and validation DataLoaders.
    """
    image_paths = []
    labels      = []

    # Class mapping
    class_map = {'no': 0, 'yes': 1}   # 0=Normal, 1=Tumor

    print(f"Loading dataset from: {data_dir}")

    for class_name, label in class_map.items():
        class_folder = os.path.join(data_dir, class_name)

        if not os.path.exists(class_folder):
            print(f"Warning: folder not found: {class_folder}")
            continue

        files = [f for f in os.listdir(class_folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"  Found {len(files)} '{class_name}' images (label={label})")

        for fname in files:
            image_paths.append(os.path.join(class_folder, fname))
            labels.append(label)

    print(f"\nTotal images: {len(image_paths)}")
    print(f"Class distribution:")
    print(f"  Normal (0): {labels.count(0)}")
    print(f"  Tumor  (1): {labels.count(1)}")

    # Split into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        test_size=val_split,
        random_state=42,
        stratify=labels       # keep class ratio balanced
    )

    print(f"\nTrain set: {len(train_paths)} images")
    print(f"Val set:   {len(val_paths)} images")

    # Create Dataset objects
    train_dataset = MedicalScanDataset(train_paths, train_labels, train_transform)
    val_dataset   = MedicalScanDataset(val_paths,   val_labels,   val_transform)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0          # keep 0 on Windows to avoid errors
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def show_dataset_samples(data_dir, num_samples=6):
    """Show sample images from the dataset."""
    import matplotlib.pyplot as plt

    class_map = {'no': 0, 'yes': 1}
    class_names = {0: 'Normal', 1: 'Tumor'}

    all_paths  = []
    all_labels = []

    for class_name, label in class_map.items():
        folder = os.path.join(data_dir, class_name)
        if not os.path.exists(folder):
            continue
        files = [f for f in os.listdir(folder)
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
        for f in files:
            all_paths.append(os.path.join(folder, f))
            all_labels.append(label)

    fig, axes = plt.subplots(1, len(all_paths), figsize=(15, 3))

    for i, (path, label) in enumerate(zip(all_paths, all_labels)):
        img = Image.open(path).convert('L')
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"{class_names[label]}", fontsize=10,
                          color='red' if label == 1 else 'green')
        axes[i].axis('off')

    plt.suptitle('Brain MRI dataset samples  |  Red = Tumor  |  Green = Normal',
                 fontsize=12)
    plt.tight_layout()
    plt.show()