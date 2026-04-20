import numpy as np

# An image is just a 2D or 3D array of numbers
# Grayscale image = 2D array (height x width)
# Color image = 3D array (height x width x 3 channels)

# Create a fake grayscale scan (100x100 pixels)
scan = np.zeros((100, 100), dtype=np.uint8)
print("Empty scan shape:", scan.shape)  # (100, 100)
print("All zeros (black):", scan[0][0])

# Simulate a bright tumor region in the middle
scan[40:60, 40:60] = 255  # white square = bright region
print("Tumor pixel value:", scan[50][50])  # 255

# Basic stats — used in real preprocessing
print("Min pixel:", np.min(scan))
print("Max pixel:", np.max(scan))
print("Mean pixel:", np.mean(scan))

# Normalize pixels to 0–1 range (very important for AI models)
normalized = scan / 255.0
print("Normalized tumor pixel:", normalized[50][50])  # 1.0