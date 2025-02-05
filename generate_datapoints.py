import numpy as np
import matplotlib.pyplot as plt
import math

def generate_and_save_gaussian_points(target_func, x_range, n_points=30, noise=100, filename="data_points"):
    """
    Generate random points around a target function with independent Gaussian noise and save them into two separate files for training and testing.

    Parameters:
    - target_func (function): The target function providing the mean values.
    - x_range (tuple): Range of the input variable (x_min, x_max).
    - n_points (int): Number of points to generate.
    - noise (float): Standard deviation of Gaussian noise.
    - filename (str): The base filename to save the generated points, without file extension.

    Returns:
    - None
    """
    # Generate evenly spaced x values in the range
    x = np.linspace(x_range[0], x_range[1], n_points).reshape(-1, 1)

    # Compute the mean values based on the target function
    y_mean = target_func(x).reshape(-1)

    # Add independent Gaussian noise to each point
    print(noise)
    actual_noise = np.random.normal(0, noise, y_mean.shape)
    print('actual_noise', actual_noise)
    y_samples = y_mean + actual_noise

    # Combine x and y to form data points
    data_points = np.column_stack((x, y_samples))

    # Shuffle the data points to ensure random distribution for train/test split
    np.random.shuffle(data_points)

    # Split the data into 80% train and 20% test
    split_index = int(0.8 * len(data_points))
    train = data_points[:split_index]
    test = data_points[split_index:]

    # Save the train and test datasets with specific names
    np.save(f"{filename}_train.npy", train)
    np.save(f"{filename}_test.npy", test)

    # Plot the points
    plt.figure(figsize=(10, 6))
    plt.plot(x, target_func(x), 'r--', label='Target Function')
    plt.scatter(train[:, 0], train[:, 1], color='green', alpha=0.6, label='Train Points')
    plt.scatter(test[:, 0], test[:, 1], color='orange', alpha=0.6, label='Test Points')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.title("Independent Gaussian Points Around Target Function")
    plt.show()

# Example usage:
if __name__ == "__main__":
    def target_function(x):
        return np.sqrt(x)   # Example target function

    x_range = (0, 5)
    n_points = 30
    filename = "Nguyen_8"
    generate_and_save_gaussian_points(target_function, x_range, n_points=n_points, noise=1000, filename=filename)

