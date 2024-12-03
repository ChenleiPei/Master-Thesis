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
from LSTM_VAE_Model import LSTMVAE, LSTMDecoder
import json


def train_gp_model(train_x, train_y):
    # since all parameters are fixed, we can use the same model and likelihood, no need for the training
    train_y = train_y.view(-1)  # Ensure train_y is a 1D tensor

    lengthscale = args.lengthscale
    outputscale = args.outputscale
    noise = args.noise
    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )
            self.covar_module.base_kernel.lengthscale = torch.tensor(
                lengthscale)  # set the lengthscale，another hyperparameter
            self.covar_module.outputscale = torch.tensor(outputscale)  # set the outputscale，another hyperparameter

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(noise)
    model = GPModel(train_x, train_y, likelihood)

    # use the evaluate mode
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
    logits, index = model.decoder(z, max_length, start_token_idx=vocab['<start>'])
    print("expression_index", index)

    # transfer the index to expression
    expression = model.reconstruct_expression(index, from_indices=True)
    print(f"Generated expression: {expression}")

    if isinstance(expression, list):
        expression = " ".join(expression)

    # transform the expression from str to math expression
    math_expression = convert_expression(expression)
    print(f"Math expression: {math_expression}")
    
    # calculate the likelihood of the math expression
    target_func = math_expression
    points_file = "datapoints.npy"
    likelihood = calculate_log_likelihood_from_gaussian(points_file, target_func, noise=0.01)

    return likelihood


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
    #exploration phase
    # should initial_position be the end position?
    #initial_position = torch.full((1, dimension), 0.5, dtype=torch.float32)
    dimension = args.dimension
    lengthscale = args.lengthscale
    outputscale = args.outputscale
    noise = args.noise
    mean = args.mean
    std_dev = args.std_dev
    steps = args.steps
    step_size = torch.full((dimension,), args.step_size, dtype=torch.float32)
    num_samples = args.num_samples
    bounds = tuple(args.bounds)
    upper_bound = args.upper_bound
    iterations = args.iterations
    number_of_test_points = args.number_of_test_points
    iterations_sampling = args.iterations_sampling

    #initial_position = torch.tensor([[0.0, 0]], dtype=torch.float32)
    initial_position = torch.tensor(args.initial_position, dtype=torch.float32)
    accumulated_samples = initial_position.clone()

    Z_score = score_func_VAE(accumulated_samples, VAEmodel, max_length, num_layers)
    # print("Z_score", Z_score)
    model, likelihood = None, None

    for i in range(iterations):

        if model is None:
            energy_function = lambda x: torch.tensor(0.0, dtype=x.dtype, device=x.device, requires_grad=True)
        else:
            energy_function = EnergyFunction(model, likelihood)

        # print("energy_function", energy_function)

        new_samples = hmc_sample(accumulated_samples[-1:], energy_function, steps, step_size, num_samples,
                                 bounds=bounds)  # use the last accepted sample as the initial position
        # print("new_samples", new_samples)
        new_samples = new_samples.view(initial_position.shape)
        # print("new_samples", new_samples)
        # print("accumulated_samples", accumulated_samples)
        accumulated_samples = torch.cat([accumulated_samples, new_samples], 0)  # Concatenate along the first dimension
        # print("accumulated_samples", accumulated_samples)
        Z_score = score_func_VAE(accumulated_samples, VAEmodel, max_length, num_layers)
        # print("Z_score", Z_score)
        print("interation", i)

        # Train the GP model with the accumulated samples
        model, likelihood = train_gp_model(accumulated_samples, Z_score)

    print("accumulated_samples", accumulated_samples)
    print("Z_score", Z_score)

    model, likelihood = model, likelihood

    # test case
    model.eval()
    likelihood.eval()

    # this test case is to test if the Gaussian Process model can predict the mean and variance of the test points
    test_points = torch.rand(number_of_test_points, dimension) * upper_bound
    print("test_points", test_points)

    # predict the mean and variance of the test points
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = model(test_points)

    # get the mean and standard deviation of the prediction
    pred_mean = pred.mean
    # pred_std = pred.stddev

    print("Predicted Mean:\n", pred_mean)
    # print("Predicted Stddev:\n", pred_std)

    real_score = score_func_VAE(test_points)
    print("Real Score:\n", real_score)

    # Calculate the RMSE
    diff = pred_mean - real_score
    squared_diff = diff ** 2
    mean_squared_diff = squared_diff.mean()
    rmse = torch.sqrt(mean_squared_diff)
    print("RMSE:\n", rmse)

    # sampling phase

    initial_position_sphase = torch.full((1, dimension), 0, dtype=torch.float32)  # Ensure this is a 2D tensor from start
    sphase_samples = initial_position_sphase.clone()

    for i in range(iterations_sampling):

        if model is None:
            energy_function = lambda x: torch.tensor(0.0, dtype=x.dtype, device=x.device, requires_grad=True)
        else:
            energy_function = EnergyFunction_sample(model, likelihood)

        # print("energy_function", energy_function)

        new_samples = hmc_sample(sphase_samples[-1:], energy_function, steps, step_size, num_samples,
                                 bounds=bounds)  # use the last accepted sample as the initial position
        # print("new_samples", new_samples)
        # new_samples = new_samples.view(-1, 1)  # Ensure new_samples is a 2D tensor, matching accumulated_samples
        new_samples = new_samples.view(initial_position.shape)
        # print("new_samples", new_samples)
        # print("accumulated_samples", accumulated_samples)
        sphase_samples = torch.cat([sphase_samples, new_samples], 0)  # Concatenate along the first dimension

        #Z_score_sphase = score_func(sphase_samples)
        print("interation_sampling", i)

    # Calculate the mean and standard deviation of the samples
    mean_dims = torch.mean(sphase_samples, dim=0)
    std_dev_dims = torch.std(sphase_samples, dim=0)

    print("sphase_samples", sphase_samples)
    print("Mean of each dimension:", mean_dims)
    print("Standard deviation of each dimension:", std_dev_dims)
    #print("Z_score_sphase", Z_score_sphase)


if __name__ == "__main__":
    #define the model with args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="LSTMVAE_bin/2024-Nov-27-17-50-32/model_final.pth")
    parser.add_argument('--vocab_path', type=str, help='Path to the vocab JSON file',
                        default='./LSTMVAE_bin/2024-Nov-27-17-50-32/vocab.json')
    parser.add_argument("--max_length", type=int, default=37)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--lengthscale", type=float, default=2)
    parser.add_argument("--outputscale", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--mean", type=float, default=10)
    parser.add_argument("--std_dev", type=float, default=1.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--step_size", type=float, default=0.1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--bounds", type=list, default=[0.0, 20.0])
    parser.add_argument("--upper_bound", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--number_of_test_points", type=int, default=50)
    parser.add_argument("--iterations_sampling", type=int, default=200)
    #define the initial position, with the defined dimension, with torch tensor and float32
    parser.add_argument("--initial_position", type=float, default=[[0, 0]])


    args = parser.parse_args()

    max_length = args.max_length
    num_layers = args.num_layers
    VAEmodel = torch.load(args.model)


    main()