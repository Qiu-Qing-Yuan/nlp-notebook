
import numpy as np
# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("Hello from server!")

# 超定系统：3 个方程，2 个未知数
A = np.array([[1, 1],
              [1, 2],
              [1, 3]])
b = np.array([1, 2, 3])

# 方法1：正规方程
x_normal = np.linalg.inv(A.T @ A) @ A.T @ b

# 方法2：np.linalg.lstsq（推荐，更稳定）
x_lstsq, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

print("正规方程解：", x_normal)
print("最小二乘解：", x_lstsq)
