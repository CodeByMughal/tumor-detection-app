
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, roc_curve)
import seaborn as sns
from ai_model.model   import get_model
from ai_model.dataset import load_brain_mri_dataset

# ── Config ────────────────────────────────────────────────────────
DATA_DIR   = 'data/brain_mri'
MODEL_PATH = 'ai_model/saved_models/brain_tumor_best.pth'
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Load model ────────────────────────────────────────────────────
model, class_names = get_model('tumor', device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Model loaded — best val acc was: {checkpoint['val_acc']:.1f}%")

# ── Load validation data ──────────────────────────────────────────
_, val_loader = load_brain_mri_dataset(DATA_DIR, batch_size=8)

# ── Collect all predictions ───────────────────────────────────────
all_preds  = []
all_labels = []
all_probs  = []

with torch.no_grad():
    for images, labels in val_loader:
        images  = images.to(device)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1)
        preds   = torch.argmax(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # tumor probability

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
all_probs  = np.array(all_probs)

# ── 1. Classification Report ──────────────────────────────────────
print("\n=== Classification Report ===")
print(classification_report(all_labels, all_preds,
                             target_names=class_names))

# ── 2. Confusion Matrix ───────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names, ax=axes[0])
axes[0].set_title('Confusion matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ── 3. ROC Curve ──────────────────────────────────────────────────
fpr, tpr, _ = roc_curve(all_labels, all_probs)
auc_score   = roc_auc_score(all_labels, all_probs)

axes[1].plot(fpr, tpr, 'b-', linewidth=2,
             label=f'ROC curve (AUC = {auc_score:.3f})')
axes[1].plot([0, 1], [0, 1], 'r--', label='Random classifier')
axes[1].set_title('ROC curve')
axes[1].set_xlabel('False positive rate')
axes[1].set_ylabel('True positive rate')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Brain tumor model evaluation', fontsize=13)
plt.tight_layout()
plt.savefig('ai_model/saved_models/brain_tumor_evaluation.png')
plt.show()

# ── Summary ───────────────────────────────────────────────────────
print(f"\n=== Summary ===")
print(f"Overall accuracy : {(all_preds == all_labels).mean()*100:.1f}%")
print(f"AUC-ROC score    : {auc_score:.3f}  (1.0 = perfect)")
tn, fp, fn, tp = cm.ravel()
print(f"True Positives   : {tp}  (correctly detected tumors)")
print(f"True Negatives   : {tn}  (correctly identified normal)")
print(f"False Positives  : {fp}  (normal labelled as tumor)")
print(f"False Negatives  : {fn}  (missed tumors ← most critical!)")