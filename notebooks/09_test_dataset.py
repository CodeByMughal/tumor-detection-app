import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ai_model.dataset import load_brain_mri_dataset, show_dataset_samples

DATA_DIR = 'data/brain_mri'

# ── Show sample images ────────────────────────────────────────────
print("=== Sample Images ===")
show_dataset_samples(DATA_DIR)

# ── Load dataset ──────────────────────────────────────────────────
print("\n=== Loading Dataset ===")
train_loader, val_loader = load_brain_mri_dataset(DATA_DIR, batch_size=8)

# ── Inspect a batch ───────────────────────────────────────────────
print("\n=== Inspecting First Batch ===")
images, labels = next(iter(train_loader))

print(f"Batch image shape: {images.shape}")
print(f"  └── {images.shape[0]} scans, "
      f"{images.shape[1]} channel, "
      f"{images.shape[2]}x{images.shape[3]} pixels")
print(f"Batch labels: {labels.tolist()}")
print(f"Pixel range: {images.min():.2f} to {images.max():.2f}")

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
print(f"\nDataset ready for training!")