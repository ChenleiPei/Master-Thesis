import argparse
import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Custom Non-Stationary RBF Kernel with Density-Based Lengthscale
class NonStationaryRBFKernelDensity(gpytorch.kernels.Kernel):
    def __init__(self, points, k=10, **kwargs):
        """
        Non-Stationary Kernel based on density.
        Args:
            points (torch.Tensor): Reference points for density calculation.
            k (int): Number of neighbors for density estimation.
        """
        super(NonStationaryRBFKernelDensity, self).__init__(has_lengthscale=False, **kwargs)
        self.points = points.cpu().numpy()  # Save reference points for density estimation
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=self.k).fit(self.points)

    def forward(self, x1, x2, diag=False, **params):
        def density_based_lengthscale(x):
            """
            Compute lengthscale based on point density.
            Args:
                x (torch.Tensor): Input points.
            Returns:
                torch.Tensor: Lengthscale for each point in x.
            """
            distances, _ = self.nn.kneighbors(x.cpu().numpy())
            avg_distance = distances.mean(axis=1)  # Average distance to k neighbors
            density = 1 / (avg_distance + 1e-5)  # Density inversely proportional to avg distance
            lengthscale = 1 / (density + 1e-5)
            print(f"Density-based lengthscales (first 10): {lengthscale[:10]}")
            return torch.clamp(torch.tensor(lengthscale, dtype=torch.float32, device=x.device), min=1e-3, max=1e3)

        lengthscale_x1 = density_based_lengthscale(x1)
        lengthscale_x2 = density_based_lengthscale(x2)

        scaled_x1 = x1 / lengthscale_x1.unsqueeze(-1)
        scaled_x2 = x2 / lengthscale_x2.unsqueeze(-1)

        diff = (scaled_x1.unsqueeze(-2) - scaled_x2.unsqueeze(-3)) ** 2
        if diag:
            return torch.exp(-0.5 * diff.sum(dim=-1).diag())
        else:
            return torch.exp(-0.5 * diff.sum(dim=-1))

# GP Model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, points, k, outputscale, noise):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            NonStationaryRBFKernelDensity(points, k=k)  # Use the density-based kernel
        )
        self.covar_module.outputscale = torch.tensor(outputscale)
        self.likelihood = likelihood
        self.likelihood.noise = torch.tensor(noise)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Uniform Sampling and GP Prediction
def uniform_sampling_and_gp_prediction(args):
    # Generate uniform grid points
    x = np.linspace(args.boundsx[0], args.boundsx[1], args.grid_size)
    y = np.linspace(args.boundsy[0], args.boundsy[1], args.grid_size)
    grid_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Simulate some valid points for GP fitting (example)
    valid_points = grid_points[:50]  # Assume first 50 are valid points
    valid_likelihoods = np.random.rand(50)  # Example likelihood values
    valid_points_tensor = torch.tensor(valid_points, dtype=torch.float32)
    likelihood_tensor = torch.tensor(valid_likelihoods, dtype=torch.float32)

    # Initialize GP model with density-based kernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = GPModel(valid_points_tensor, likelihood_tensor, likelihood,
                    points=valid_points_tensor, k=args.k_neighbors,
                    outputscale=args.outputscale, noise=args.noise)

    # Set the model to evaluation mode for prediction
    model.eval()
    likelihood.eval()

    # Visualize the GP model predictions
    visualize_gp_model(model, grid_points, valid_points, grid_points_tensor, likelihood_tensor)

    return model, likelihood

# Visualize GP Model Predictions
def visualize_gp_model(model, grid_points, valid_points, grid_points_tensor, likelihoods):
    x = np.linspace(-5, 15, 100)
    y = np.linspace(-5, 10, 100)
    X, Y = np.meshgrid(x, y)
    visualization_points = np.vstack([X.flatten(), Y.flatten()]).T
    visualization_tensor = torch.tensor(visualization_points, dtype=torch.float32)

    mean = None
    if model:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = model(visualization_tensor)
            mean = predictions.mean.cpu().numpy().reshape(X.shape)
            print("Mean Prediction - Min:", mean.min(), "Max:", mean.max())

            # Clip extreme values and normalize
            mean = np.clip(mean, -1e6, 1e6)
            mean = (mean - mean.min()) / (mean.max() - mean.min() + 1e-6)

    plt.figure(figsize=(20, 16))
    if mean is not None:
        contour = plt.contourf(X, Y, mean, levels=100, cmap="coolwarm", alpha=0.7)
        cbar = plt.colorbar(contour, pad=0.1)
        cbar.set_label("Mean Prediction", fontsize=14)

    # Plot valid points
    plt.scatter(valid_points[:, 0], valid_points[:, 1], color="red", label="Valid Points", edgecolors="k")

    # Plot all grid points
    plt.scatter(grid_points_tensor[:, 0], grid_points_tensor[:, 1], color="blue", alpha=0.2, label="Grid Points")

    plt.title("GP Model Predictions (Density-Based Lengthscale)", fontsize=16)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_size", type=int, default=50, help="Number of grid points per axis")
    parser.add_argument("--lengthscale", type=float, default=2.0, help="Default lengthscale (unused in density kernel)")
    parser.add_argument("--outputscale", type=float, default=1.0, help="Kernel output scale")
    parser.add_argument("--noise", type=float, default=0.01, help="Observation noise")
    parser.add_argument("--boundsx", type=list, default=[-5, 15], help="Bounds for x-axis")
    parser.add_argument("--boundsy", type=list, default=[-5, 10], help="Bounds for y-axis")
    parser.add_argument("--k_neighbors", type=int, default=10, help="Number of neighbors for density estimation")
    args = parser.parse_args()

    uniform_sampling_and_gp_prediction(args)
