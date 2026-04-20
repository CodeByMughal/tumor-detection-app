import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import models
import torch.nn as nn
from ai_model.gradcam import predict_with_gradcam

device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Normal', 'Tumor']

# ── Load trained MobileNet model ──────────────────────────────────
def load_mobilenet(model_path, device):
    model = models.mobilenet_v2(weights=None)

    # Match the architecture from training
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 2)
    )
    model.features[0][0] = nn.Conv2d(
        1, 32, kernel_size=3, stride=2, padding=1, bias=False
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Model loaded — val acc: {checkpoint['val_acc']:.1f}%")
    return model


MODEL_PATH = 'ai_model/saved_models/brain_tumor_mobilenet_best.pth'
model      = load_mobilenet(MODEL_PATH, device)

# ── Test on tumor scans ───────────────────────────────────────────
print("\n=== Testing on TUMOR scans ===")
tumor_folder = 'data/brain_mri/yes'
tumor_images = [f for f in os.listdir(tumor_folder)
                if f.lower().endswith(('.jpg','.jpeg','.png'))][:3]

for fname in tumor_images:
    path   = os.path.join(tumor_folder, fname)
    save   = f'ai_model/saved_models/gradcam_tumor_{fname}'
    result = predict_with_gradcam(
        model, path, CLASS_NAMES,
        device=device, save_path=save
    )
    print(f"{fname}: {result['prediction']} "
          f"({result['confidence']:.1f}%)")

# ── Test on normal scans ──────────────────────────────────────────
print("\n=== Testing on NORMAL scans ===")
normal_folder = 'data/brain_mri/no'
normal_images = [f for f in os.listdir(normal_folder)
                 if f.lower().endswith(('.jpg','.jpeg','.png'))][:2]

for fname in normal_images:
    path   = os.path.join(normal_folder, fname)
    save   = f'ai_model/saved_models/gradcam_normal_{fname}'
    result = predict_with_gradcam(
        model, path, CLASS_NAMES,
        device=device, save_path=save
    )
    print(f"{fname}: {result['prediction']} "
          f"({result['confidence']:.1f}%)")

print("\nAll Grad-CAM images saved to ai_model/saved_models/")
print("Phase 2 complete!")