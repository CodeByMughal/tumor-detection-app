import torch
import torch.nn as nn

# ── Build the simplest possible neural network ────────────────────
# Input: a flattened 224x224 image = 50176 numbers
# Output: 2 classes (Tumor / Normal)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(50176, 512),   # layer 1: 50176 inputs → 512 neurons
            nn.ReLU(),               # activation: adds non-linearity
            nn.Linear(512, 128),     # layer 2: 512 → 128 neurons
            nn.ReLU(),
            nn.Linear(128, 2)        # output: 2 classes (Tumor / Normal)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)   # flatten image to 1D
        return self.network(x)


model = SimpleNet()
print("=== Model Architecture ===")
print(model)

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ── Simulate a forward pass ───────────────────────────────────────
# Pretend we have 4 scans in a batch
fake_batch = torch.zeros(4, 1, 224, 224)
output = model(fake_batch)

print(f"\nInput shape:  {fake_batch.shape}")
print(f"Output shape: {output.shape}")
print(f"Output (raw scores):\n{output}")

# ── Convert raw scores to probabilities ───────────────────────────
probabilities = torch.softmax(output, dim=1)
print(f"\nProbabilities:\n{probabilities}")

# ── Get predicted class for each scan ─────────────────────────────
predictions = torch.argmax(probabilities, dim=1)
classes = ['Normal', 'Tumor']
print("\nPredictions:")
for i, pred in enumerate(predictions):
    print(f"  Scan {i+1}: {classes[pred.item()]} "
          f"({probabilities[i][pred].item()*100:.1f}% confidence)")