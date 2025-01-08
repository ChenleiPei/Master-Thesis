import torch
from matplotlib import pyplot as plt
#import pymc3 as pm
#import theano.tensor as tt

from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import normalize, unnormalize
from botorch.models.transforms import Standardize, Normalize
from botorch.optim import optimize_acqf


def score_func(z):

    with torch.no_grad():

        return 50 * torch.cos(z * torch.sqrt(0.1 * z)) + 0.01 * (z - 2.5) * (z - 1) * (z - 5) * (z-19) * (z - 19.5)


def plot_score_func():
    zz = torch.linspace(0, 20, steps=200)
    yy = score_func(zz)

    plt.plot(zz, yy, label='Ground Truth', color='blue')  # 添加label和指定颜色

def plot_gp_posterior(gp, bounds):
    zz = torch.linspace(bounds[0].item(), bounds[-1].item(), steps=200)
    post = gp.posterior(normalize(zz, bounds))
    mean = torch.squeeze(post.mean.detach())
    std = torch.sqrt(torch.squeeze(post.variance.detach()))

    plt.fill_between(zz, mean - std, mean + std, alpha=0.2, color='orange')  # 指定填充颜色
    plt.plot(zz, mean, label='Posterior', color='orange')  # 添加label和指定颜色

    plt.xlabel('Input X')
    plt.ylabel('Output Y')
    plt.legend()
    plt.show()


def scatter_data(Z):
    Z_score = score_func(Z)
    plt.scatter(Z, Z_score, label='test points')


def get_maximum_of_acquisition_func(acq_func, bounds):

    z_new, _ = optimize_acqf(
        acq_func,
        bounds=torch.as_tensor([[0.0],[1.0]]),
        q=1,
        num_restarts=10,
        raw_samples=256
    )

    z_new = unnormalize(z_new, bounds=bounds)

    return z_new, score_func(z_new)



def main():

    Z = torch.as_tensor([0.0, 20], dtype=torch.float64)
    Z = torch.unsqueeze(Z, dim=-1)
    Z_score = score_func(Z)

    bounds = torch.as_tensor([[0], [20]])
    gp = SingleTaskGP(train_X=normalize(Z, bounds), train_Y=Z_score, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    ucb = UpperConfidenceBound(gp, beta=1.0)

    plt.figure()
    plot_score_func()
    scatter_data(Z)
    plot_gp_posterior(gp, bounds)
    plt.show()


    for i in range(10):
        z_new, z_score = get_maximum_of_acquisition_func(ucb, bounds)

        Z = torch.cat([Z, z_new])
        Z_score = torch.cat([Z_score, z_score])

        gp = SingleTaskGP(train_X=normalize(Z, bounds), train_Y=Z_score, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        ucb = UpperConfidenceBound(gp, beta=1.0)

        plt.figure()
        plot_score_func()
        scatter_data(Z)
        plot_gp_posterior(gp, bounds)
        plt.show()


if __name__ == "__main__":
    main()
