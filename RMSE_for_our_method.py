import numpy as np

def load_xy_values(xy_values_file):
    data = np.load(xy_values_file)
    x_values = data[:, 0]  
    y_true = data[:, 1]   
    return x_values, y_true

def evaluate_equation(equation, x):
   
    return eval(equation)

def calculate_rmse(equation, x_values, y_true):
    y_pred = [evaluate_equation(equation, x) for x in x_values]
    rmse = np.sqrt(((y_pred - y_true) ** 2).mean())
    return rmse


equation1 = "x*x*x + x*x + x"



xy_values_file = 'Koza_3_test.npy'

x_values, y_true = load_xy_values(xy_values_file)
rmse1 = calculate_rmse(equation1, x_values, y_true)
print("RMSE value for the equation:", rmse1)




