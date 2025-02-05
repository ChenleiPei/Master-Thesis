import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# define the model function
def model_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# datapoints
x = np.linspace(0, 4, 50)
y = model_func(x, 2.5, 1.3, 0.5) + 0.2 * np.random.normal(size=x.size)

# use curve fit to estimate the parameters
popt, pcov = curve_fit(model_func, x, y, p0=[2, 1, 0.5])


plt.figure()
plt.scatter(x, y, label='Data')
plt.plot(x, model_func(x, *popt), 'r-', label='Fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
plt.legend()
plt.show()
