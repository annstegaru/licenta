import torch
import numpy as np

#x = torch.ones(2 ,2, dtype = torch.int) #1 = size
#x = torch.tensor([2.5, 0.1])

x = torch.rand(4,4)
#print(x)
#print(x[:,0]) #prima col
# print second row
#print(x[1, :])

#print(x[1, 1].item())

# print(x)
# y = x.view(-1, 8)
# print(y.size())

# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# a.add_(1)
# print(a)
# print(b)

# a = np.ones(5)
# print(a)
# b = torch.from_numpy(a)
# print(b)

# a += 1
# print(a)
# print(b)

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.ones(5, device = device)
#     y = torch.ones(5) 
#     y = y.to(device)
#     z = x + y
#     z = z.to("cpu")


# x = torch.ones(5, requires_grad=True)
# print(x)

x = torch.randn(3, requires_grad=True)
print(x)

# y = x+2
# print(y)
# z = y*y*2
# # z = z.mean()
# print(z)

# v = torch.tensor([0.1, 1.0, 0.001], dtype = torch.float32)
# z.backward(v) # dz/dx
# # print(x.grad)

# x.requires_grad_(False)
# x.detach()
# with torch.no_grad():
with torch.no_grad():
    y = x + 2
    print(y)