import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    Highlights which regions of a scan the model
    focused on when making its prediction.
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None

        # Register hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap for input scan."""
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if none specified
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, class_idx].backward()

        # Compute weights — global average of gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)

        # Weighted combination of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)   # only positive influence

        # Normalize to 0-1
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_idx


def apply_heatmap(original_img, cam, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on original scan.
    Returns a colored image with tumor region highlighted.
    """
    # Resize cam to match image size
    h, w = original_img.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    # Convert to colormap (COLORMAP_JET: blue→green→red)
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Convert original to RGB if grayscale
    if len(original_img.shape) == 2:
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = original_img.copy()

    # Blend heatmap with original
    overlay = cv2.addWeighted(original_rgb, 1 - alpha,
                               heatmap, alpha, 0)
    return overlay, heatmap


def get_gradcam_for_mobilenet(model):
    """Get the right target layer for MobileNetV2."""
    # Last conv layer in MobileNetV2 feature extractor
    return model.features[-1][0]


def predict_with_gradcam(model, image_path, class_names,
                          device='cpu', save_path=None):
    """
    Full pipeline:
    Load scan → preprocess → predict → generate heatmap → visualize
    """
    # ── Load and preprocess image ─────────────────────────────────
    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    original_display = cv2.resize(original, (224, 224))

    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    pil_img      = Image.open(image_path).convert('RGB')
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    # ── Get prediction ────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs  = torch.softmax(output, dim=1)

    pred_class = probs.argmax(dim=1).item()
    confidence = probs[0][pred_class].item() * 100
    label      = class_names[pred_class]

    # ── Generate Grad-CAM ─────────────────────────────────────────
    # Re-run with gradients enabled
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    target_layer = get_gradcam_for_mobilenet(model)
    gradcam      = GradCAM(model, target_layer)
    cam, _       = gradcam.generate(input_tensor, class_idx=pred_class)

    overlay, heatmap = apply_heatmap(original_display, cam, alpha=0.45)

    # ── Visualize ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(original_display, cmap='gray')
    axes[0].set_title('Original scan', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM heatmap\n(red = model focus)', fontsize=11)
    axes[1].axis('off')

    axes[2].imshow(overlay)
    color = 'red' if pred_class == 1 else 'green'
    axes[2].set_title(f'Overlay\nPrediction: {label} ({confidence:.1f}%)',
                      fontsize=11, color=color)
    axes[2].axis('off')

    plt.suptitle(
        f'Grad-CAM Analysis  |  {label}  |  Confidence: {confidence:.1f}%',
        fontsize=13, color=color
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

    return {
        'prediction': label,
        'confidence': confidence,
        'class_idx':  pred_class,
        'cam':        cam
    }