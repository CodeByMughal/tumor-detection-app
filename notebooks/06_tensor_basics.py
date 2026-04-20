import torch
import numpy as np

print("=== PyTorch version:", torch.__version__)
print("=== GPU available:", torch.cuda.is_available())

# ── 1. Creating tensors ───────────────────────────────────────────
t1 = torch.tensor([1.0, 2.0, 3.0])
print("\nSimple tensor:", t1)
print("Shape:", t1.shape)
print("Data type:", t1.dtype)

# ── 2. Tensors that look like images ─────────────────────────────
# A single grayscale scan = (1 channel, 224 height, 224 width)
single_scan = torch.zeros(1, 224, 224)
print("\nSingle scan shape:", single_scan.shape)

# A batch of 8 scans = (8 batch, 1 channel, 224, 224)
batch_of_scans = torch.zeros(8, 1, 224, 224)
print("Batch of 8 scans shape:", batch_of_scans.shape)

# ── 3. Tensor operations (same as NumPy but for AI) ───────────────
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

print("\nMatrix multiply (used in every neural network layer):")
print(torch.matmul(a, b))

print("\nElement-wise multiply:")
print(a * b)

# ── 4. Convert NumPy array to Tensor (used in preprocessing) ──────
numpy_scan = np.zeros((1, 224, 224), dtype=np.float32)
tensor_scan = torch.from_numpy(numpy_scan)
print("\nNumPy → Tensor shape:", tensor_scan.shape)

# ── 5. Gradients — the magic behind AI learning ───────────────────
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1   # y = x² + 3x + 1

y.backward()   # compute gradient

print("\nGradient of y=x²+3x+1 at x=2:")
print("Expected: 2*2 + 3 = 7")
print("Got:     ", x.grad.item())