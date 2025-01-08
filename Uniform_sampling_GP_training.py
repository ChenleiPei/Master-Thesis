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
        self.mean_module.constant = train_y.mean()  # Set the mean to the average of the training data
        print(f"Initial Mean value: {self.mean_module.constant}")

        # Use Matern kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)
        self.covar_module.outputscale = torch.tensor(outputscale)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class EnergyFunction(torch.nn.Module):
    def __init__(self, model, likelihood):
        super(EnergyFunction, self).__init__()
        self.model = model
        self.likelihood = likelihood

    def forward(self, x):
        self.model.eval()
        self.likelihood.eval()

        with torch.enable_grad():
            predictive_distribution = self.likelihood(self.model(x))
            mean = predictive_distribution.mean
            std = predictive_distribution.stddev

            # energy function (from the paper)
            energy = mean - std

        return energy

class EnergyFunction_sample(torch.nn.Module):
    def __init__(self, model, likelihood):
        super(EnergyFunction_sample, self).__init__()
        self.model = model
        self.likelihood = likelihood

    def forward(self, x):
        self.model.eval()
        self.likelihood.eval()

        with torch.enable_grad():
            predictive_distribution = self.likelihood(self.model(x))
            mean = predictive_distribution.mean
            std = predictive_distribution.stddev

            # energy function (from the paper)
            energy = mean

        return energy

def hmc_sample(initial_position, energy_function, steps, step_size, num_samples, bounds):
    samples = []
    position = initial_position.clone().detach().requires_grad_(True)
    # print("initial_position", initial_position)

    for _ in range(num_samples):
        # initialize the momentum
        momentum = torch.randn_like(position)
        current_position = position.clone()
        current_position.requires_grad_(True)

        for _ in range(steps):
            energy = energy_function(position)
            # print("energy", energy)
            # print("position", position)

            gradient = torch.autograd.grad(energy, position, create_graph=True, allow_unused=True)[0]

            if gradient is None:
                gradient = torch.zeros_like(position)

            # print("gradient", gradient)
            # print("momentum", momentum)

            # update momentum
            # print("momentum", momentum)
            # print("gradient", gradient)
            momentum = momentum - (step_size * gradient / 2)
            # print("new momentum", momentum)

            # update position
            position = position + (step_size * momentum)

            # constrain the position to be within the bounds
            position = torch.clamp(position, *bounds)

            # calculate the gradient of the energy function using torch.autograd.grad
            # print("position", position)
            energy = energy_function(position)
            gradient = torch.autograd.grad(energy, position, create_graph=True, allow_unused=True)[0]
            if gradient is None:
                gradient = torch.zeros_like(position)
            # print("new gradient", gradient)
            momentum = momentum - (step_size * gradient / 2)
            position.requires_grad_(True)

        # calculate the energy for the current position and the proposed position
        current_energy = energy_function(current_position)
        proposed_energy = energy_function(position)
        acceptance_prob = torch.exp(current_energy - proposed_energy).item()

        # accept or reject the new position
        if torch.rand(1).item() < acceptance_prob:
            position = position.clone()
            samples.append(position.clone().detach())

        samples = torch.stack(samples) if samples else initial_position.unsqueeze(0)

    return samples

def train_gp_model(train_x, train_y, args):
    print(f"Train x: {train_x}, Train y: {train_y}")
    train_y = train_y.view(-1)

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(args.noise)

    model = GPModel(train_x, train_y, likelihood, args.lengthscale, args.outputscale, args.noise)

    # Training settings
    """model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=0.2)

    # Use the marginal log likelihood (MLL) loss
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Training loop
    num_training_iterations = 1000  # Number of iterations for training
    for i in range(num_training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)  # Loss is negative MLL
        loss.backward()
        print(f"Iteration {i + 1}/{num_training_iterations} - Loss: {loss.item():.4f}")
        optimizer.step()"""

    # Set the model and likelihood to evaluation mode after training
    model.eval()
    likelihood.eval()

    return model, likelihood

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
        result = score_func_VAE(z, VAEmodel, max_length, vocab, args)
        if result is not None:
            likelihood, expression = result
            likelihood_list.append(likelihood)
            valid_points.append(z.tolist())
            expressions.append(expression)
        else:
            failed_points.append(z.tolist())
            expressions.append(None)

    # convert to tensor
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

def score_func_VAE(z, model, max_length, vocab, args):
    logits, index = model.decoder(z.unsqueeze(0), max_length, start_token_idx=vocab['<start>'])

    if index.dim() == 1:
        index = index.unsqueeze(0)

    print(f"Index shape before reconstruct_expression: {index.shape}")

    expressions = model.reconstruct_expression(index, from_indices=True)
    expression = expressions[0] if len(expressions) > 0 else None

    if expression:
        print(f"Reconstructed expression: {expression}")
    else:
        print(f"No valid expression reconstructed from index: {index}")

    try:
        math_expression = convert_expression(expression)
        points_file = "datapoints_g4x+2.npy"
        likelihood = calculate_log_likelihood_from_gaussian(points_file, math_expression, std=args.std_likelihood)
        return likelihood, expression
    except Exception as e:
        print(f"Error with latent point {z.tolist()}: {e}")
        return None

def score_func_VAE(z, model, max_length, num_layers):
    # Calculate the score function using a VAE model.
    # first decode the latent position z to get the expression
    with open(args.vocab_path, 'r') as f:
        vocab = json.load(f)
    print("z", z)

    # Generate expression using the decoder
    logits, index = model.decoder(z, max_length, start_token_idx=vocab['<start>'])
    print("expression_index", index)

    # Transfer the index to expression
    expression = model.reconstruct_expression(index, from_indices=True)
    print(f"Generated expression: {expression}")

    if isinstance(expression, list):
        expression = " ".join(expression)

    try:
        # Transform the expression from str to math expression
        math_expression = convert_expression(expression)
        print(f"Math expression: {math_expression}")

        # Calculate the likelihood of the math expression
        target_func = math_expression
        points_file = "datapoints_g4x+2.npy"
        likelihood = calculate_log_likelihood_from_gaussian(points_file, target_func, std=args.std_likelihood)
        print(f"Likelihood: {likelihood}")

        '''# Check if the generated expression is meaningful (e.g., valid mathematical expression)
        if math_expression is None or not math_expression:
            print(f"Skipping invalid expression: {expression}")
            return None  # Skip this iteration if the expression is not meaningful'''

    except Exception as e:
        print(f"Error converting expression {expression}: {e}")
        return None  # Skip this iteration if there's an error in expression conversion


    return likelihood, math_expression

def visualize_gp_model(model, grid_points, valid_points, failed_points, expressions, likelihoods):
    x = np.linspace(-5, 15, args.grid_size)
    y = np.linspace(-5, 8, args.grid_size)
    X, Y = np.meshgrid(x, y)
    visualization_points = np.vstack([X.flatten(), Y.flatten()]).T

    visualization_tensor = torch.tensor(visualization_points, dtype=torch.float32)
    mean = None

    if model:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = model(visualization_tensor)
            mean = predictions.mean.numpy().reshape(X.shape)
            mean = np.clip(mean, -1e6, 1e6)
            mean = (mean - mean.min()) / (mean.max() - mean.min() + 1e-6)
            mean = np.log(mean + 1)

    unique_expressions = list(set(filter(None, expressions)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_expressions)))
    markers = ['o', 's', '^', 'D', 'P', '*', 'X', 'v', 'h', '+']
    expression_to_style = {exp: (colors[i], markers[i % len(markers)]) for i, exp in enumerate(unique_expressions)}

    plt.figure(figsize=(20, 16))

    if mean is not None:
        contour = plt.contourf(X, Y, mean, levels=100, cmap="coolwarm", alpha=0.7)
        cbar = plt.colorbar(contour, pad=0.1)
        cbar.set_label("Mean Prediction", fontsize=14)

    for point, expression in zip(grid_points, expressions):
        if expression is None:
            plt.scatter(point[0], point[1], color="lightgray", edgecolors="k", alpha=0.5)
        else:
            color, marker = expression_to_style[expression]
            plt.scatter(point[0], point[1], color=color, marker=marker, edgecolors="k", alpha=0.8)

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

    if mean is not None:
        fig = plt.figure(figsize=(20, 16))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, mean, cmap="viridis", alpha=0.8)
        ax.set_zlim(mean.min(), mean.max())
        ax.set_title("3D Surface of GP Mean Prediction", fontsize=16)
        ax.set_xlabel("X", fontsize=14)
        ax.set_ylabel("Y", fontsize=14)
        ax.set_zlabel("Mean Prediction", fontsize=14)

    plt.show()


def main():
    # exploration phase
    dimension = args.dimension
    lengthscale = args.lengthscale
    outputscale = args.outputscale
    std = args.std_likelihood
    steps = args.steps
    step_size = torch.full((dimension,), args.step_size, dtype=torch.float32)
    num_samples = args.num_samples
    #bounds = tuple(args.bounds)
    #set bounds with boundx and boundy
    bounds = torch.tensor([args.boundsx, args.boundsy], dtype=torch.float32)
    iterations = args.iterations

    initial_position = torch.tensor(args.initial_position, dtype=torch.float32)

    accumulated_samples = initial_position.clone()
    trained_iterations = []
    accumulated_samples_list = []  # for saving the accumulated samples
    math_expressions_list = []
    model, likelihood = model, likelihood = uniform_sampling_and_gp_fitting(args)
    #dimension = args.dimension
    Z_score = torch.empty((1), dtype=torch.float32)
    #print the shape of Z_score
    print("shape of Z_score", Z_score.shape)

    # Initial model and likelihood creation
    result_first = score_func_VAE(accumulated_samples, VAEmodel, max_length, num_layers, args)

    if result_first is not None:
        Z_score, expression = result_first
        Z_score = torch.tensor(Z_score, dtype=torch.float32).view(1)

        model, likelihood = train_gp_model(accumulated_samples, Z_score, args)
        accumulated_samples_list.append(initial_position.tolist()[0])  # 保存样本，确保是 1x2 的格式
        math_expressions_list.append(expression)

    # for saving the math expressions

    for i in range(iterations):
        if model is None:
            energy_function = lambda x: torch.tensor(0.0, dtype=x.dtype, device=x.device, requires_grad=True)
        else:
            energy_function = EnergyFunction(model, likelihood)

        new_samples = hmc_sample(accumulated_samples[-1:], energy_function, steps, step_size, num_samples,
                                 bounds=bounds)  # use the last accepted sample as the initial position
        new_samples = new_samples.view(initial_position.shape)

        result = score_func_VAE(new_samples, VAEmodel, max_length, num_layers)

        if result is not None:
            Z_score_new, expression = result
            Z_score_new = torch.tensor(Z_score_new, dtype=torch.float32).view(1)
            print("shape of Z_score_new", Z_score_new.shape)
            Z_score = torch.cat([Z_score, Z_score_new], 0)

            # Ensure new_samples has shape (1, dimension) and convert it to a list
            new_samples = new_samples.view(1, -1)  # Ensure it has two elements per point (e.g., [x, y])
            accumulated_samples = torch.cat([accumulated_samples, new_samples], 0)

            accumulated_samples_list.append(new_samples.tolist()[0])  # with shape (2,)
            math_expressions_list.append(expression)  # Save the math expressions

            # Train the GP model with the accumulated samples if new samples were added
            model, likelihood = train_gp_model(accumulated_samples, Z_score, args)

            trained_iterations.append(i)

        print("accumulated_samples", accumulated_samples)
        print("Z_score", Z_score)
        print("iteration", i)
        print("trained_iterations", trained_iterations)

    # save the accumulated samples and math expressions as JSON files
    with open('accumulated_samples.json', 'w') as f:
        json.dump(accumulated_samples_list, f, indent=4)

    # save the math expressions as a JSON file
    math_expressions_list_str = [str(expr) for expr in math_expressions_list]

    with open('math_expressions.json', 'w') as f:
        json.dump(math_expressions_list_str, f, indent=4)

    # Visualization
    visualize_gp_model(model, accumulated_samples_list, math_expressions_list_str)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="LSTMVAE_bin/2024-Dec-04-19-08-42/model_final.pth")
    parser.add_argument('--vocab_path', type=str, default='./LSTMVAE_bin/2024-Dec-04-19-08-42/vocab.json')
    parser.add_argument("--max_length", type=int, default=37)
    parser.add_argument("--grid_size", type=int, default=50, help="Number of points per axis for uniform sampling")
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--lengthscale", type=float, default=1)
    parser.add_argument("--outputscale", type=float, default=5.0)
    parser.add_argument("--std_likelihood", type=float, default=1)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--boundsx", type=list, default=[-5, 15])
    parser.add_argument("--boundsy", type=list, default=[-5, 8])
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step_size", type=float, default=0.015)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--number_of_test_points", type=int, default=50)
    parser.add_argument("--iterations_sampling", type=int, default=200)
    parser.add_argument("--initial_position", type=float, default=[[0, 0]])

    args = parser.parse_args()
    max_length = args.max_length
    num_layers = args.num_layers
    VAEmodel = torch.load(args.model)

    main()

