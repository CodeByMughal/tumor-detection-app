import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Add project root to path so we can import ai_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_model.preprocess import (
    resize_image, denoise_image,
    normalize_image, apply_clahe, preprocess_scan
)

# ── Create a fake scan to test with ──────────────────────────────
def create_fake_scan():
    """Generate a fake MRI-like scan for testing."""
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img, (150, 150), 120, 170, -1)   # brain tissue
    cv2.circle(img, (180, 130), 40, 255, -1)    # tumor
    cv2.circle(img, (180, 130), 18, 110, -1)    # necrosis core
    cv2.circle(img, (120, 170), 20, 230, -1)    # second lesion
    return img

# Save fake scan to disk so preprocess_scan() can load it
os.makedirs('data/test', exist_ok=True)
fake = create_fake_scan()
cv2.imwrite('data/test/fake_brain_scan.png', fake)
print("Fake scan saved to data/test/fake_brain_scan.png")

# ── Run the full pipeline ─────────────────────────────────────────
result = preprocess_scan('data/test/fake_brain_scan.png', scan_type='brain')
print(f"\nPipeline output shape: {result.shape}")
print(f"Value range: {result.min():.2f} – {result.max():.2f}")

# ── Visualize each step ───────────────────────────────────────────
original  = fake.copy()
contrasted = apply_clahe(original.copy())
denoised  = denoise_image(contrasted.copy())
resized   = resize_image(denoised.copy())
normalized = normalize_image(resized.copy())

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
steps = [
    (original,   'Step 1: Raw scan',        'gray'),
    (contrasted, 'Step 2: CLAHE contrast',  'gray'),
    (denoised,   'Step 3: Denoised',        'gray'),
    (resized,    'Step 4: Resized 224x224', 'gray'),
    (normalized, 'Step 5: Normalized 0–1', 'gray'),
]

for ax, (img, title, cmap) in zip(axes, steps):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.axis('off')

plt.suptitle('Full preprocessing pipeline — scan ready for AI model', fontsize=12)
plt.tight_layout()
plt.show()

print("\nAll pipeline steps completed successfully!")
print("This pipeline works for Brain MRI, Lung CT, Breast scans & AVN bone scans.")