import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Create a fake MRI-like scan ---
scan = np.zeros((300, 300), dtype=np.uint8)

# Background brain tissue (gray)
cv2.circle(scan, (150, 150), 120, 180, -1)

# Tumor region (bright white)
cv2.circle(scan, (170, 130), 35, 255, -1)

# Necrosis inside tumor (darker core)
cv2.circle(scan, (170, 130), 15, 120, -1)

# --- Basic preprocessing steps ---

# 1. Resize — AI models need fixed input size
resized = cv2.resize(scan, (224, 224))
print("Original size:", scan.shape)
print("Resized to:", resized.shape)

# 2. Normalize to 0-1
normalized = resized / 255.0

# 3. Add Gaussian blur — removes noise from real scans
blurred = cv2.GaussianBlur(resized, (5, 5), 0)

# 4. Edge detection — helps highlight tumor boundaries
edges = cv2.Canny(resized, 50, 150)

# --- Show all 4 versions ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(scan, cmap='gray')
axes[0].set_title('Original scan')
axes[0].axis('off')

axes[1].imshow(resized, cmap='gray')
axes[1].set_title('Resized (224x224)')
axes[1].axis('off')

axes[2].imshow(blurred, cmap='gray')
axes[2].set_title('Denoised (blur)')
axes[2].axis('off')

axes[3].imshow(edges, cmap='gray')
axes[3].set_title('Edge detection')
axes[3].axis('off')

plt.suptitle('Medical scan preprocessing pipeline', fontsize=13)
plt.tight_layout()
plt.show()