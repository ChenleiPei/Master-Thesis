import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel


def train_gp_model(train_x, train_y, training_iter=100, lr=0.1):
    train_y = train_y.view(-1)  # Ensure train_y is a 1D tensor

    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(GPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(lengthscale_constraint=gpytorch.constraints.Interval(1.0, 1.1)))

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.LessThan(0.1),
    )
    likelihood.noise = torch.tensor(0.01)  # It is set the start value, but will be updated during training
    model = GPModel(train_x, train_y, likelihood)

    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            _ + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()

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

        # 这里启用梯度计算
        with torch.enable_grad():
            # 获得预测的后验分布
            predictive_distribution = self.likelihood(self.model(x))

            # 计算均值和标准差
            mean = predictive_distribution.mean
            std = predictive_distribution.stddev

            # 定义能量方程
            energy = mean - std

        return energy



def score_func(z):
    with torch.no_grad():
        # return 50 * torch.cos(z * torch.sqrt(0.1 * z)) + 0.01 * (z - 2.5) * (z - 1) * (z - 5) * (z - 19) * (z - 19.5)
        # return (z - 10) * torch.sin(z) + 10 * torch.cos(z) + 0.1 * z * z
        return (z - 10) * torch.sin(z) + torch.cos(z) + 0.1 * z * z


def plot_score_func(ax):
    zz = torch.linspace(0, 20, steps=200).unsqueeze(-1)
    yy = score_func(zz)
    ax.plot(zz.numpy(), yy.numpy(), label="Score Function", color='black')
    ax.set_xlabel("z")
    ax.set_ylabel("Score")
    ax.set_title("Score Function and GP Posterior")


def plot_gp_posterior(model, likelihood, ax, bounds):
    test_x = torch.linspace(bounds[0], bounds[1], 100)
    with torch.no_grad():
        observed_pred = likelihood(model(test_x.unsqueeze(-1)))
        mean = observed_pred.mean
        std = observed_pred.stddev
        # lower, upper = observed_pred.confidence_region()
        lower, upper = mean - 3 * std, mean + 3 * std
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_title("GP Posterior")


def hmc_sample(initial_position, energy_function, steps=50, step_size=0.01, num_samples=1, bounds=(0.0, 20.0)):
    samples = []
    position = initial_position.clone().detach().requires_grad_(True)

    for _ in range(num_samples):
        momentum = torch.randn_like(position)
        current_position = position.clone()
        current_position.requires_grad_(True)

        for _ in range(steps):
            energy = energy_function(position)
            # calculate the gradient of the energy function using torch.autograd.grad
            # gradient = torch.autograd.grad(energy, position, create_graph=True, allow_unused=True)[0]
            gradient = torch.autograd.grad(energy_function(position), position, create_graph=True,
                                           allow_unused=True)[0]

            if gradient is None:
                gradient = torch.zeros_like(position)

            print(gradient)

            momentum = momentum - (step_size * gradient / 2)
            position = position + (step_size * momentum)
            position = torch.clamp(position, *bounds)
            position.requires_grad_(True)

            # energy = energy_function(position)
            # gradient = grad(energy, position, create_graph=True, allow_unused=True)[0]
            # if gradient is None:
            # gradient = torch.zeros_like(position)

            # momentum = momentum - (step_size * gradient / 2)

        current_energy = energy_function(current_position)
        proposed_energy = energy_function(position)
        acceptance_prob = torch.exp(current_energy - proposed_energy).item()

        if torch.rand(1) < acceptance_prob:
            current_position = position
            samples.append(position.detach())
        else:
            current_position = current_position

    return torch.stack(samples) if samples else initial_position.unsqueeze(0)


def main():
    initial_position = torch.tensor([[0.0], [20.0]], dtype=torch.float32)  # Ensure this is a 2D tensor from start
    accumulated_samples = initial_position.clone()
    bounds = (0.0, 20.0)
    model, likelihood = None, None
    #energy_function = EnergyFunction(model, likelihood)

    for i in range(10):  # Let's do 5 rounds of sampling and training

        if model is None:
            energy_function = lambda x: torch.tensor(0.0, dtype=x.dtype, device=x.device, requires_grad=True)
        else:
            energy_function = EnergyFunction(model, likelihood)

        new_samples = hmc_sample(accumulated_samples[-1:], energy_function, steps=50, step_size=0.05, num_samples=1,
                                 bounds=bounds)#use the last accepted sample as the initial position
        new_samples = new_samples.view(-1, 1)  # Ensure new_samples is a 2D tensor, matching accumulated_samples

        accumulated_samples = torch.cat([accumulated_samples, new_samples], 0)  # Concatenate along the first dimension
        Z_score = score_func(accumulated_samples)

        # Train the GP model with the accumulated samples
        model, likelihood = train_gp_model(accumulated_samples, Z_score)


        fig, ax = plt.subplots()
        plot_score_func(ax)
        plot_gp_posterior(model, likelihood, ax, bounds)
        ax.scatter(accumulated_samples[:, 0].numpy(), Z_score.numpy(), color='red')
        plt.show()


if __name__ == "__main__":
    main()
