import numpy as np

a = np.load("tunnel/output/depth_npy/0001.npy")
b = np.load("001.npy")

print("shape:", a.shape, b.shape)
print("dtype:", a.dtype, b.dtype)

num_diff = np.sum(a != b)

print("不同元素数:", num_diff)
print("总元素数:", a.size)