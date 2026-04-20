import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load a sample image (we'll use a built-in test image for now)
img = np.zeros((300, 300, 3), dtype=np.uint8)

# Draw a white circle (simulating a tumor region)
cv2.circle(img, center=(150, 150), radius=60, color=(255, 255, 255), thickness=-1)

# Draw a smaller darker circle inside (simulating necrosis)
cv2.circle(img, center=(150, 150), radius=25, color=(100, 100, 100), thickness=-1)

# Display it
plt.figure(figsize=(5, 5))
plt.imshow(img, cmap='gray')
plt.title("Simulated scan — tumor region")
plt.axis('off')
plt.show()

print("Image shape:", img.shape)
print("Setup complete! You are ready for Phase 2.")