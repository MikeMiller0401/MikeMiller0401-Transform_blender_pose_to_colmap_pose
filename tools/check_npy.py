import numpy as np

a = np.load("tunnel/output/merged_pcd.npy")
b = np.load("tunnel/output/new_merged_pcd.npy")

print("shape:", a.shape, b.shape)
print("dtype:", a.dtype, b.dtype)

num_diff = np.sum(a != b)

print("不同元素数:", num_diff)
print("总元素数:", a.size)