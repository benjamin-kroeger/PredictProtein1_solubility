import torch

# Enable CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create tensors and set requires_grad=True
x = torch.randn(1000, 1000, device=device, requires_grad=True)
y = torch.randn(1000, 1000, device=device, requires_grad=True)
z = torch.matmul(x, y)

# Create a gradient tensor with the same shape as z
gradient = torch.ones_like(z)

# Calculate gradients
gradients = torch.autograd.grad(z, [x, y], gradient)

# Calculate memory occupied by gradients
gradient_memory = sum([grad.element_size() * grad.nelement() for grad in gradients])

# Print memory occupied by gradients
print("Memory occupied by gradients: {:.2f} MB".format(gradient_memory / 1024 / 1024))

# Print maximum memory occupied by gradients
print("Maximum memory occupied by gradients: {:.2f} MB".format(torch.cuda.max_memory_allocated() / 1024 / 1024))
