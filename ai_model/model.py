import torch
import torch.nn as nn

class TumorCNN(nn.Module):
    """
    Custom CNN for medical scan analysis.
    Works for: Brain MRI, Lung CT, Breast mammography, AVN bone scans.
    Input:  (batch, 1, 224, 224) — grayscale scan
    Output: class scores (Tumor/Normal or AVN stages)
    """

    def __init__(self, num_classes=2):
        super(TumorCNN, self).__init__()

        # ── Feature Extractor ─────────────────────────────────────
        # This part learns WHAT to look for in the scan
        self.features = nn.Sequential(

            # Block 1 — detect basic edges and shapes
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32 filters
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 224 → 112

            # Block 2 — detect complex patterns
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 filters
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 112 → 56

            # Block 3 — detect tumor-specific features
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 128 filters
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 56 → 28

            # Block 4 — high-level feature detection
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 28 → 14
        )

        # ── Classifier ────────────────────────────────────────────
        # This part decides WHICH CLASS based on detected features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),    # → (256, 4, 4)
            nn.Flatten(),                    # → 4096
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Dropout(0.5),                 # prevent overfitting
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)      # final output
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(task='tumor', device='cpu'):
    """
    Get the right model configuration for each scan task.

    task options:
      'tumor'  — binary: Tumor vs Normal (brain, lung, breast)
      'avn'    — 5 classes: Normal + AVN Stage 1/2/3/4
    """
    if task == 'tumor':
        num_classes = 2
        class_names = ['Normal', 'Tumor']
    elif task == 'avn':
        num_classes = 5
        class_names = ['Normal', 'AVN Stage 1', 'AVN Stage 2',
                       'AVN Stage 3', 'AVN Stage 4']
    else:
        raise ValueError(f"Unknown task: {task}")

    model = TumorCNN(num_classes=num_classes)
    model = model.to(device)

    return model, class_names