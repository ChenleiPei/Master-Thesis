import numpy as np
import matplotlib.pyplot as plt

# Define five mathematical expressions

def expr1(x):
    return 4 * x + 2

def expr2(x):
    return 2 * 2 * x + 2

def expr3(x):
    return 2 * np.sin(x)

def expr4(x):
    return 2 + np.cos(np.sin(x * x))

def expr5(x):
    return np.exp(3 / np.exp(2 + x / 2) + x)

# Set the range for x
x = np.linspace(0.1, 10, 400)  # Avoid x=0 to prevent division by zero

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the five expressions
plt.plot(x, expr1(x), label=r'$4 * x + 2$', color='b')
plt.plot(x, expr2(x), label=r'$2 * 2 * x + 2$', color='g')
plt.plot(x, expr3(x), label=r'$2 * \sin(x)$', color='r')
plt.plot(x, expr4(x), label=r'$2 + \cos(\sin(x * x))$', color='c')
#plt.plot(x, expr5(x), label=r'$\exp\left(\frac{3}{\exp(2 + \frac{x}{2})} + x\right)$', color='m')

# Generate Gaussian points around a target function
def generate_and_save_gaussian_points(target_func, x_range, n_points=10, noise=0.01, filename="data_points.npy"):
    # Generate evenly spaced x values in the range
    x = np.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)

    # Compute the mean values based on the target function
    y_mean = target_func(x).reshape(-1)

    # Add independent Gaussian noise to each point
    y_samples = y_mean + np.random.normal(0, noise, y_mean.shape)

    # Save the points
    np.save(filename, np.column_stack((x, y_samples)))

    # Plot the points
    plt.scatter(x, y_samples, color='blue', alpha=0.6, label='Generated Gaussian Points')

    return x, y_samples

# Define the target function
def target_function(x):
    return 4 * x + 2

# Generate and plot Gaussian points
x_range = (0.1, 10)
n_points = 10
filename = "datapoints.npy"
generate_and_save_gaussian_points(target_function, x_range, n_points=n_points, noise=1, filename=filename)

# Add labels and title
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of Five Mathematical Expressions with Gaussian Points')

# Display the legend
plt.legend()

# Display the grid
plt.grid(True)

# Show the plot
plt.show()
