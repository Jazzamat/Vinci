from re import L
import torch
x = torch.rand(5, 3)
print(x)


if (torch.cuda.is_available()):
    print("GPU driver and CUDA is enabled")
else:
    print("GPU driver and CUDA is NOT enabled")