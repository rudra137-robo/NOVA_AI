
import torch

print("PyTorch version:", torch.__version__)
print("Device:", "CUDA GPU" if torch.cuda.is_available() else "CPU")
print("Random tensor:")
print(torch.randn(3, 3))

# A tiny learning experiment
x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[2.0], [4.0], [6.0]])

weight = torch.nn.Parameter(torch.randn(1, 1))
optimizer = torch.optim.SGD([weight], lr=0.01)

for step in range(200):
    prediction = x @ weight
    loss = ((prediction - y) ** 2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Learned weight:", weight.item())
print("Final loss:", loss.item())
