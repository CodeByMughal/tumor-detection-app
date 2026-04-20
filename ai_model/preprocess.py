import cv2
import numpy as np
import os

# Standard input size for all AI models in this project
IMG_SIZE = 224

def load_image(path, grayscale=True):
    """Load an image from disk."""
    if grayscale:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")

    return img


def resize_image(img, size=IMG_SIZE):
    """Resize to standard model input size."""
    return cv2.resize(img, (size, size))


def denoise_image(img):
    """Remove noise using Gaussian blur."""
    return cv2.GaussianBlur(img, (5, 5), 0)


def normalize_image(img):
    """Normalize pixel values to 0.0 – 1.0 range."""
    return img.astype(np.float32) / 255.0


def apply_clahe(img):
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization.
    Enhances contrast in MRI/CT scans — makes tumors more visible.
    Used in real medical imaging AI systems.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def preprocess_scan(path, scan_type='brain'):
    """
    Full pipeline — takes raw scan path, returns model-ready array.
    Works for brain, lung, breast, and bone (AVN) scans.
    """
    print(f"Processing {scan_type} scan: {path}")

    # Step 1: Load
    img = load_image(path, grayscale=True)
    print(f"  Loaded: {img.shape}")

    # Step 2: Enhance contrast (especially useful for MRI/CT)
    img = apply_clahe(img)
    print(f"  Contrast enhanced")

    # Step 3: Denoise
    img = denoise_image(img)
    print(f"  Denoised")

    # Step 4: Resize to 224x224
    img = resize_image(img)
    print(f"  Resized to: {img.shape}")

    # Step 5: Normalize to 0-1
    img = normalize_image(img)
    print(f"  Normalized — min: {img.min():.2f}, max: {img.max():.2f}")

    # Step 6: Add channel dimension for PyTorch (1 x 224 x 224)
    img = np.expand_dims(img, axis=0)
    print(f"  Final shape: {img.shape}")

    return img


def batch_preprocess(folder_path, scan_type='brain'):
    """
    Process an entire folder of scans at once.
    Returns list of processed arrays + filenames.
    """
    results = []
    supported = ('.jpg', '.jpeg', '.png', '.bmp')

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(supported)]

    print(f"Found {len(files)} images in {folder_path}")

    for fname in files:
        full_path = os.path.join(folder_path, fname)
        try:
            processed = preprocess_scan(full_path, scan_type)
            results.append({'filename': fname, 'data': processed})
        except Exception as e:
            print(f"  Skipped {fname}: {e}")

    return results