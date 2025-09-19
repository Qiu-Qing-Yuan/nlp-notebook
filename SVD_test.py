import numpy as np
A = np.array([[1, 1], [0, 1]])
U, Sigma, VT = np.linalg.svd(A)
print("U:", U)
print("Sigma:", Sigma)  # 注意：返回的是奇异值数组，不是完整矩阵
print("V^T:", VT)