import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
from torch.distributions import Normal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from scipy.stats import norm
from utils import  convert_expression
from calculate_likelihood import calculate_log_likelihood_from_gaussian
import argparse
#from LSTM_VAE_Model import LSTMVAE, LSTMDecoder
import json
import torch
import gpytorch
from ac_grammar_vae.model.gvae import GrammarVariationalAutoencoder

# define the GP model
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale, outputscale, noise):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )
        self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale)  # set the lengthscale
        self.covar_module.outputscale = torch.tensor(outputscale)  # set the outputscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gp_model(train_x, train_y, args):
    # Ensure train_y is a 1D tensor
    train_y = train_y.view(-1)

    lengthscale = args.lengthscale
    outputscale = args.outputscale
    noise = args.noise

    # Initialize likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(noise)

    # Initialize model
    model = GPModel(train_x, train_y, likelihood, lengthscale, outputscale, noise)

    # Use evaluation mod
    model.eval()
    likelihood.eval()

    return model, likelihood



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
        points_file = args.points_file
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


def main():
    # exploration phase
    dimension = args.dimension
    lengthscale = args.lengthscale
    outputscale = args.outputscale
    std = args.std_likelihood
    steps = args.steps
    step_size = torch.full((dimension,), args.step_size, dtype=torch.float32)
    num_samples = args.num_samples
    bounds = tuple(args.bounds)
    iterations = args.iterations

    initial_position = torch.tensor(args.initial_position, dtype=torch.float32)

    accumulated_samples = initial_position.clone()
    trained_iterations = []
    accumulated_samples_list = []  # for saving the accumulated samples
    math_expressions_list = []
    model, likelihood = None, None
    #dimension = args.dimension
    Z_score = torch.empty((1), dtype=torch.float32)
    #print the shape of Z_score
    print("shape of Z_score", Z_score.shape)

    # Initial model and likelihood creation
    result_first = score_func_VAE(accumulated_samples, VAEmodel, max_length, num_layers)

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

            accumulated_samples_list.append(new_samples.tolist()[0])  # 保存样本，确保是 1x2 的格式
            math_expressions_list.append(expression)  # 保存数学表达式


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


# Visualization
def visualize_gp_model(model, accumulated_samples_list, math_expressions_list):
    # 创建较小网格数据进行预测，减少内存需求
    x = np.linspace(-5, 15, 20)
    y = np.linspace(-5, 15, 20)
    X, Y = np.meshgrid(x, y)

    # 将网格数据展平并转化为 PyTorch 张量，确保其形状为 (M, 2)
    inputs = np.vstack([X.flatten(), Y.flatten()]).T
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)


    batch_size = 50
    mean_list = []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for i in range(0, len(inputs_tensor), batch_size):
            batch_inputs = inputs_tensor[i:i + batch_size]
            pred = model(batch_inputs)
            mean_list.append(pred.mean.numpy())


    mean = np.concatenate(mean_list)


    Z = mean.reshape(X.shape)

    import matplotlib.cm as cm

    unique_expressions = list(set(math_expressions_list))
    colors = cm.get_cmap('tab10')

    expression_to_color = {exp: colors(i / len(unique_expressions)) for i, exp in enumerate(unique_expressions)}

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)

    for sample, expression in zip(accumulated_samples_list, math_expressions_list):
        point = np.array(sample)
        if len(point) >= 2:
            #let z be the lowest position
            z_bottom = Z.min()
            ax.scatter(point[0], point[1], z_bottom, color=expression_to_color[expression], s=20, alpha=0.8)


    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=expression_to_color[exp], markersize=10)
               for exp in unique_expressions]
    ax.legend(handles, unique_expressions, loc='upper right', bbox_to_anchor=(1.3, 1))


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Mean Prediction')

    ax.set_title('3D Visualization of GP Model Mean Predictions with Accumulated Samples')

    plt.show()


if __name__ == "__main__":
    # define the model with args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LSTMVAE_bin/2024-Dec-04-19-08-42/model_final.pth")
    parser.add_argument('--vocab_path', type=str, help='Path to the vocab JSON file',
                        default='./LSTMVAE_bin/2024-Dec-04-19-08-42/vocab.json')
    parser.add_argument("--max_length", type=int, default=37)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--lengthscale", type=float, default=2)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--std_likelihood", type=float, default=1)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--mean", type=float, default=10)
    parser.add_argument("--std_dev", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--step_size", type=float, default=0.015)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--bounds", type=list, default=[-3, 15])
    parser.add_argument("--upper_bound", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--number_of_test_points", type=int, default=50)
    parser.add_argument("--iterations_sampling", type=int, default=200)
    parser.add_argument("--initial_position", type=float, default=[[0, 0]])
    parser.add_argument("--points_file", type=str, default="datapoints_g4x+2.npy")

    args = parser.parse_args()
    max_length = args.max_length
    num_layers = args.num_layers
    VAEmodel = torch.load(args.model)

    model = main()

    # save the model
    torch.save(model.state_dict(), "GPmodel_state_dict.pth")

