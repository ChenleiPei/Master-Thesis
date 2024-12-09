import numpy as np
import matplotlib.pyplot as plt


def generate_and_save_gaussian_points(target_func, x_range, n_points=10, noise=0.01, filename="data_points.npy"):
    """
    Generate random points around a target function with independent Gaussian noise and save them.

    Parameters:
    - target_func (function): The target function providing the mean values.
    - x_range (tuple): Range of the input variable (x_min, x_max).
    - n_points (int): Number of points to generate.
    - noise (float): Standard deviation of Gaussian noise.
    - filename (str): The filename to save the generated points.

    Returns:
    - x (numpy array): Input values.
    - y (numpy array): Output values around the target function.
    """
    # Generate evenly spaced x values in the range
    x = np.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)

    # Compute the mean values based on the target function
    y_mean = target_func(x).reshape(-1)

    # Add independent Gaussian noise to each point
    y_samples = y_mean + np.random.normal(0, noise, y_mean.shape)

    # Save the points
    np.save(filename, np.column_stack((x, y_samples)))

    # Plot the points
    plt.figure(figsize=(10, 6))
    plt.plot(x, target_func(x), 'r--', label='Target Function')
    plt.scatter(x, y_samples, color='blue', alpha=0.6, label='Generated Gaussian Points')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Independent Gaussian Points Around Target Function")
    plt.show()

    return x, y_samples


# Example usage:
if __name__ == "__main__":
    # Define the target function as a Python function
    def target_function(x):
        return (4*x + 2)  # Example: sin(x)


    # Generate and save Gaussian points around the target function
    x_range = (0.1, 10)
    n_points = 10
    filename = "datapoints.npy"
    generate_and_save_gaussian_points(target_function, x_range, n_points=n_points, noise=1, filename=filename)
