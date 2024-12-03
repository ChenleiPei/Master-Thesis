import numpy as np
import matplotlib.pyplot as plt

# 定义三个数学表达式
def expr1(x):
    return (2 + (1 + 2) / np.cos(np.cos(x)))

def expr2(x):
    return np.sin(np.sin(1)) / (2 + x * np.sin(2 + 1 / 1) + np.sin(x / x))

def expr3(x):
    return x + np.cos(np.exp(1 / x * np.sin(3 / x) + x))

# 设置 x 范围
x = np.linspace(0.1, 10, 400)  # 避免 x=0 时除以零的问题

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制三个表达式
plt.plot(x, expr1(x), label=r'$2 + (1 + 2) / \cos(\cos(x))$', color='b')
plt.plot(x, expr2(x), label=r'$\frac{\sin(\sin(1))}{2 + x \cdot \sin(2 + \frac{1}{1}) + \sin(\frac{x}{x})}$', color='r')
plt.plot(x, expr3(x), label=r'$x + \cos(\exp(\frac{1}{x} \cdot \sin(\frac{3}{x}) + x))$', color='g')

# 添加标签和标题
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of Three Mathematical Expressions')

# 显示图例
plt.legend()

# 显示网格
plt.grid(True)

# 显示图形
plt.show()
