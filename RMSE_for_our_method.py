import numpy as np

def load_xy_values(xy_values_file):
    data = np.load(xy_values_file)
    x_values = data[:, 0]  # 假设x值在每行的第一列
    y_true = data[:, 1]    # 假设y值在每行的第二列
    return x_values, y_true

def evaluate_equation(equation, x):
    # 使用 eval 计算等式的值，这里假设等式是正确的 Python 表达式
    return eval(equation)

def calculate_rmse(equation, x_values, y_true):
    y_pred = [evaluate_equation(equation, x) for x in x_values]
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    return rmse

# 手动输入等式
equation1 = "x*x*x + x*x + x"


# 文件路径，需要根据实际情况进行修改
xy_values_file = 'Koza_3_test.npy'

x_values, y_true = load_xy_values(xy_values_file)
rmse1 = calculate_rmse(equation1, x_values, y_true)
print("RMSE value for the equation:", rmse1)




