import matplotlib.pyplot as plt
import numpy as np

# 模拟复杂度轴
complexity = np.linspace(0.1, 5, 100)

# 模拟偏差²（随复杂度增加而减少）
bias_sq = 4 * np.exp(-complexity)

# 模拟方差（随复杂度增加而增加）
variance = 0.5 * (np.exp(complexity) - 1)

# 总误差 = 偏差² + 方差 + 噪声（假设噪声=0.5）
total_error = bias_sq + variance + 0.5

# 绘图
plt.figure(figsize=(10, 6))
plt.plot(complexity, bias_sq, label='Bias²', color='blue', linewidth=2)
plt.plot(complexity, variance, label='Variance', color='red', linewidth=2)
plt.plot(complexity, total_error, label='Total Error (Test Error)', color='green', linewidth=3)

plt.xlabel('Model Complexity', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.title('Bias-Variance Tradeoff', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 8)
plt.show()

# 真实值和预测值
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# 计算 SSE
SSE = np.sum((y_true - y_pred) ** 2)
print("SSE =", SSE)  # 输出：SSE = 1.25 + 0.25 + 0 + 1 = 2.5

# 对比 MSE
MSE = np.mean((y_true - y_pred) ** 2)
print("MSE =", MSE)  # 输出：MSE = 2.5 / 4 = 0.625