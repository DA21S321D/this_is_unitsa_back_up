import torch

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available")
    # Create a tensor and move it to GPU
    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print("Tensor on GPU:", x)
else:
    print("CUDA is not available")
