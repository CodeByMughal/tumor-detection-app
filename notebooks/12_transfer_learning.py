import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'     
import sys, os  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ai_model.transfer_model import get_resnet50, get_mobilenet
from ai_model.dataset        import load_brain_mri_dataset
from ai_model.train          import train_model

DATA_DIR = 'data/brain_mri'
device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

train_loader, val_loader = load_brain_mri_dataset(DATA_DIR, batch_size=8)

# ── Option A: MobileNet (faster on CPU ~5 mins) ───────────────────
print("\nTraining with MobileNetV2...")
mobile_model = get_mobilenet(num_classes=2, device=device)

history, best_acc = train_model(
    model         = mobile_model,
    train_loader  = train_loader,
    val_loader    = val_loader,
    num_epochs    = 15,
    learning_rate = 0.001,
    device        = device,
    model_name    = 'brain_tumor_mobilenet'
)

print(f"\nMobileNet best accuracy: {best_acc:.1f}%")
print("Compare this to our custom CNN accuracy of 80.4%")
print("Transfer learning should give 88–95%!")