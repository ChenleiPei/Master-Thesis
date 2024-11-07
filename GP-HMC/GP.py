import torch
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
from torch.distributions import Normal
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel
from scipy.stats import norm

#parameters
#train_gp_model
lengthscale = 2.0
outputscale = 1.0
noise = 0.01


def train_gp_model(train_x, train_y):
    #since all parameters are fixed, we can use the same model and likelihood, no need for the training
    train_y = train_y.view(-1)  # Ensure train_y is a 1D tensor
    class GPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
                super(GPModel, self).__init__(train_x, train_y, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )
                self.covar_module.base_kernel.lengthscale = torch.tensor(lengthscale) #set the lengthscale，another hyperparameter
                self.covar_module.outputscale = torch.tensor(outputscale) #set the outputscale，another hyperparameter

        def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    likelihood.noise = torch.tensor(noise)
    model = GPModel(train_x, train_y, likelihood)

    #use the evaluate mode
    model.eval()
    likelihood.eval()

    return model, likelihood