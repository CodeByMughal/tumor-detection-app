import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ai_model.model   import get_model
from ai_model.dataset import load_brain_mri_dataset
from ai_model.train   import train_model

# ── Config ────────────────────────────────────────────────────────
DATA_DIR      = 'data/brain_mri'
BATCH_SIZE    = 8
NUM_EPOCHS    = 20
LEARNING_RATE = 0.001
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Device: {device}")

# ── Load dataset ──────────────────────────────────────────────────
print("\nLoading dataset...")
train_loader, val_loader = load_brain_mri_dataset(
    DATA_DIR, batch_size=BATCH_SIZE
)

# ── Get model ─────────────────────────────────────────────────────
model, class_names = get_model('tumor', device)
print(f"Model classes: {class_names}")

# ── Train ─────────────────────────────────────────────────────────
history, best_acc = train_model(
    model        = model,
    train_loader = train_loader,
    val_loader   = val_loader,
    num_epochs   = NUM_EPOCHS,
    learning_rate= LEARNING_RATE,
    device       = device,
    model_name   = 'brain_tumor'
)

print(f"\nFinal best accuracy: {best_acc:.1f}%")
print("Brain MRI model training complete!")