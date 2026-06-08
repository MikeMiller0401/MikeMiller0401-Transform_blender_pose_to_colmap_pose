import numpy as np

a = np.load("tunnel/output/pointcloud_single/0001.npy")
b = np.load("tunnel/output/pointclouds/0001.npy")

print("shape:", a.shape, b.shape)
print("dtype:", a.dtype, b.dtype)

num_diff = np.sum(a != b)

print("不同元素数:", num_diff)
print("总元素数:", a.size)