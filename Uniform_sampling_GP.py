import torch
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import json
from utils import convert_expression
from calculate_likelihood import calculate_log_likelihood_from_gaussian
from LSTM_VAE_Model import LSTMVAE
import argparse

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale, outputscale, noise):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        self.covar_module.outputscale = torch.tensor(outputscale)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def uniform_sampling_and_gp_fitting(args):
    # generate grid points
    x = np.linspace(args.boundsx[0], args.boundsx[1], args.grid_size)
    y = np.linspace(args.boundsy[0], args.boundsy[1], args.grid_size)
    grid_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # load VAE model and vocab
    VAEmodel = torch.load(args.model)
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)

    max_length = args.max_length
    likelihood_list = []
    valid_points = []
    failed_points = []
    expressions = []

    # for each grid point, calculate the likelihood
    for z in grid_points_tensor:
        result = score_function_vae(z, VAEmodel, max_length, vocab, args)
        if result is not None:
            likelihood, expression = result
            likelihood_list.append(likelihood)
            valid_points.append(z.tolist())
            expressions.append(expression)
        else:
            failed_points.append(z.tolist())
            expressions.append(None)

    # convert to tensor
    #print the valid points and corresponding likelihood one to one
    #for i in range(len(valid_points)):
        #print(f"Valid point: {valid_points[i]}, Likelihood: {likelihood_list[i]}")
    valid_points_tensor = torch.tensor(valid_points, dtype=torch.float32) if valid_points else torch.empty((0, 2))
    likelihood_tensor = torch.tensor(likelihood_list, dtype=torch.float32) if likelihood_list else torch.empty((0,))

    # GP model fitting
    if valid_points:
        model, likelihood = train_gp_model(valid_points_tensor, likelihood_tensor, args)
    else:
        model, likelihood = None, None

    # visualize the GP model fit
    visualize_gp_model(model, grid_points, valid_points, failed_points, expressions, likelihood_tensor)

    return model, likelihood

# VAE的Score Function
def score_function_vae(z, model, max_length, vocab, args):
    logits, index = model.decoder(z.unsqueeze(0), max_length, start_token_idx=vocab['<start>'])

    # 确保 index 是二维张量，即使 batch_size 为 1
    if index.dim() == 1:  # 如果是一维张量，扩展为 (1, seq_length)
        index = index.unsqueeze(0)

    print(f"Index shape before reconstruct_expression: {index.shape}")

    # 调用 reconstruct_expression
    expressions = model.reconstruct_expression(index, from_indices=True)

    # 假设 batch_size 为 1，取第一个表达式
    expression = expressions[0] if len(expressions) > 0 else None

    if expression:
        print(f"Reconstructed expression: {expression}")
    else:
        print(f"No valid expression reconstructed from index: {index}")

    try:
        # 转换为数学表达式
        math_expression = convert_expression(expression)

        # 计算likelihood
        points_file = "datapoints.npy"  # 数据点文件路径
        likelihood = calculate_log_likelihood_from_gaussian(points_file, math_expression, std=args.std_likelihood)
        return likelihood, expression
    except Exception as e:
        print(f"Error with latent point {z.tolist()}: {e}")
        return None


# visualize the GP model fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import gpytorch

from mpl_toolkits.mplot3d import Axes3D

def visualize_gp_model(model, grid_points, valid_points, failed_points, expressions, likelihoods):
    # Create grid for visualization
    x = np.linspace(-5, 15, 100)
    y = np.linspace(-5, 15, 100)
    X, Y = np.meshgrid(x, y)
    visualization_points = np.vstack([X.flatten(), Y.flatten()]).T  # Shape (10000, 2)

    visualization_tensor = torch.tensor(visualization_points, dtype=torch.float32)

    # GP predictions
    mean = None
    if model:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = model(visualization_tensor)
            mean = predictions.mean.numpy().reshape(X.shape)  # Ensure it matches grid shape
            #use log of the mean
            #mean = np.log(mean)
            #mean = np.log(mean + 1)
            #mean = np.log1p(mean)  # 等价于 np.log(1 + mean)

            #mean = (mean - np.min(mean)) / (np.max(mean) - np.min(mean) + 1e-6)  # 归一化到 [0, 1]
            #mean = np.log(mean + 1e-4)

    # Unique expressions for styling
    unique_expressions = list(set(filter(None, expressions)))  # Exclude None
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_expressions)))
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', 'h', '+']
    expression_to_style = {exp: (colors[i], markers[i % len(markers)]) for i, exp in enumerate(unique_expressions)}

    # Create a 2D visualization
    plt.figure(figsize=(20, 16))

    # If GP model exists, draw contour
    if mean is not None:
        contour = plt.contourf(X, Y, mean, levels=100, cmap="viridis", alpha=0.7)
        cbar = plt.colorbar(contour, pad=0.1)  # Adjust color bar position
        cbar.set_label("Mean Prediction", fontsize=14)

    # Plot grid points with expressions
    for point, expression in zip(grid_points, expressions):
        if expression is None:  # Failed points
            plt.scatter(point[0], point[1], color="lightgray", edgecolors="k", alpha=0.5)
        else:  # Successful points
            color, marker = expression_to_style[expression]
            plt.scatter(point[0], point[1], color=color, marker=marker, edgecolors="k", alpha=0.8)

    # Add legend for expressions
    handles = [
        plt.Line2D([0], [0], marker=marker, color=color, linestyle="", markersize=10)
        for exp, (color, marker) in expression_to_style.items()
    ]
    plt.legend(
        handles,
        unique_expressions,
        loc="upper left",
        bbox_to_anchor=(1.25, 1),
        fontsize=12,
        title="Expressions",
        title_fontsize=14
    )

    plt.title("GP Model Fit to Uniformly Sampled Points", fontsize=16)
    plt.xlabel("X", fontsize=14)
    plt.ylabel("Y", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # Create a 3D plot for GP mean
    if mean is not None:
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, mean, cmap="viridis", alpha=0.8)

        # Add labels and title
        ax.set_title("3D Surface of GP Mean Prediction", fontsize=16)
        ax.set_xlabel("X", fontsize=14)
        ax.set_ylabel("Y", fontsize=14)
        ax.set_zlabel("Mean Prediction", fontsize=14)

    plt.show()



def train_gp_model(train_x, train_y, args):
    train_y = train_y.view(-1)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(args.noise)

    model = GPModel(train_x, train_y, likelihood, args.lengthscale, args.outputscale, args.noise)
    model.eval()
    likelihood.eval()

    return model, likelihood

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LSTMVAE_bin/2024-Dec-04-19-08-42/model_final.pth")
    parser.add_argument('--vocab_path', type=str, default='./LSTMVAE_bin/2024-Dec-04-19-08-42/vocab.json')
    parser.add_argument("--max_length", type=int, default=37)
    parser.add_argument("--grid_size", type=int, default=50, help="Number of points per axis for uniform sampling")
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--lengthscale", type=float, default=2)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--std_likelihood", type=float, default=1)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--boundsx", type=list, default=[-5, 15])
    parser.add_argument("--boundsy", type=list, default=[-5, 10])

    args = parser.parse_args()
    model, likelihood = uniform_sampling_and_gp_fitting(args)

