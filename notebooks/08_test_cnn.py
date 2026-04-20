import torch
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_model.model import TumorCNN, get_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ── Test 1: Tumor detection model ─────────────────────────────────
print("\n=== Tumor Detection Model ===")
tumor_model, tumor_classes = get_model('tumor', device)
print(f"Classes: {tumor_classes}")

# Count parameters
params = sum(p.numel() for p in tumor_model.parameters())
print(f"Total parameters: {params:,}")

# Forward pass with fake batch of 4 scans
fake_scans = torch.zeros(4, 1, 224, 224).to(device)
output = tumor_model(fake_scans)
probs  = torch.softmax(output, dim=1)
preds  = torch.argmax(probs, dim=1)

print("\nTumor model predictions on fake scans:")
for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = tumor_classes[pred.item()]
    conf  = prob[pred].item() * 100
    print(f"  Scan {i+1}: {label} ({conf:.1f}% confidence)")

# ── Test 2: AVN detection model ───────────────────────────────────
print("\n=== AVN Bone Disease Model ===")
avn_model, avn_classes = get_model('avn', device)
print(f"Classes: {avn_classes}")

params = sum(p.numel() for p in avn_model.parameters())
print(f"Total parameters: {params:,}")

output = avn_model(fake_scans)
probs  = torch.softmax(output, dim=1)
preds  = torch.argmax(probs, dim=1)

print("\nAVN model predictions on fake scans:")
for i, (pred, prob) in enumerate(zip(preds, probs)):
    label = avn_classes[pred.item()]
    conf  = prob[pred].item() * 100
    print(f"  Scan {i+1}: {label} ({conf:.1f}% confidence)")

# ── Test 3: Check the layers ──────────────────────────────────────
print("\n=== CNN Layer by Layer ===")
print(tumor_model)

print("\nAll CNN tests passed!")
print("Your CNN is ready to be trained on real medical scans.")