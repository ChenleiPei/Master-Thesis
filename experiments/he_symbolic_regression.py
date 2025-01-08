import torch

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import mlflow
from functools import partial

from matplotlib import pyplot as plt
from ac_grammar_vae.model.gvae.interpreter import TorchEquationInterpreter, ExpressionWithParameters

from ac_grammar_vae.config.experiment import register_isotherm_config, SymbolicIsothermExperimentConfig
from ac_grammar_vae.data.sorption.problem import SymbolicIsothermProblem


def plot_expression(expression, interpreter=None, domain=(-5.0, 5.0), steps=250, data=None):

    plt.figure()
    xx = torch.linspace(*domain, steps=steps)

    if interpreter is not None:
        yy = interpreter.evaluate(expression, x=xx)
    else:
        yy = expression(xx)

    plt.plot(xx.numpy(), yy.numpy())
    plt.title("".join(expression) if interpreter is not None else str(expression))

    if data is not None:
        plt.scatter(data[0], data[1])


def ogden_model(lambda_1, lambda_2, lambda_3, alphas = [], mus = []):
    energy = 0.0
    for alpha, mu in zip(alphas, mus):
        energy += mu / alpha * (lambda_1**alpha + lambda_2**alpha + lambda_3**alpha - 3.0)
    return energy


ogden_mat1 = partial(ogden_model, alphas=[2.5, 0.5], mus=[.01, .24])
ogden_mat2 = partial(ogden_model, alphas=[2.5, 0.5], mus=[.15, .24])
ogden_mat3 = partial(ogden_model, alphas=[0.75, -0.25, 2.89], mus=[.1, .03, .005])


@hydra.main(version_base="1.2", config_path="../config", config_name="he_symbolic_regression")
def main(cfg: DictConfig):

    n_opt_steps = cfg.n_opt_steps

    if cfg.problem_name == "ogden-1":
        current_model = ogden_mat1

    if cfg.problem_name == "ogden-2":
        current_model = ogden_mat2

    if cfg.problem_name == "ogden-3":
        current_model = ogden_mat3

    n_data = 20
    n_test = 500
    X_train = {
        '\\lambda_1': torch.unsqueeze(2.0 + torch.rand(n_data), dim=-1),
        '\\lambda_2': torch.unsqueeze(2.0 + torch.rand(n_data), dim=-1),
        '\\lambda_3': torch.unsqueeze(2.0 + torch.rand(n_data), dim=-1),
    }
    Y_train = current_model(X_train['\\lambda_1'], X_train['\\lambda_2'], X_train['\\lambda_3'])

    X_test1 = {
        '\\lambda_1': torch.unsqueeze(2.0 + torch.rand(n_test), dim=-1),
        '\\lambda_2': torch.unsqueeze(2.0 + torch.rand(n_test), dim=-1),
        '\\lambda_3': torch.unsqueeze(2.0 + torch.rand(n_test), dim=-1),
    }
    Y_test1 = current_model(X_test1['\\lambda_1'], X_test1['\\lambda_2'], X_test1['\\lambda_3'])

    X_test2 = {
        '\\lambda_1': torch.unsqueeze(0.75 + torch.rand(n_test), dim=-1),
        '\\lambda_2': torch.unsqueeze(0.75 + torch.rand(n_test), dim=-1),
        '\\lambda_3': torch.unsqueeze(0.75 + torch.rand(n_test), dim=-1),
    }
    Y_test2 = current_model(X_test2['\\lambda_1'], X_test2['\\lambda_2'], X_test2['\\lambda_3'])

    X_test3 = {
        '\\lambda_1': torch.unsqueeze(3.25 + torch.rand(n_test), dim=-1),
        '\\lambda_2': torch.unsqueeze(3.25 + torch.rand(n_test), dim=-1),
        '\\lambda_3': torch.unsqueeze(3.25 + torch.rand(n_test), dim=-1),
    }
    Y_test3 = current_model(X_test3['\\lambda_1'], X_test3['\\lambda_2'], X_test3['\\lambda_3'])

    # load the model from file
    model = torch.load(to_absolute_path("results/gvae_pretrained_parametric_3.pth"))

    mlflow.set_tracking_uri(to_absolute_path("./mlruns"))
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():

        # log the parameters
        mlflow.log_param("problem", cfg.problem_name)
        mlflow.log_param("n_opt_steps", cfg.n_opt_steps)

        # fit the expression
        torch.random.seed()
        model.eval()
        expr = model.find_expression_for(X=X_train, Y=Y_train, num_opt_steps=n_opt_steps)

        rmse = torch.sqrt(torch.mean(torch.square(Y_train - expr(X_train))))
        mlflow.log_metric("training/RMSE", rmse.item())

        # compute the metrics
        for name, dataset in {"test1": (X_test1, Y_test1), "test2": (X_test2, Y_test2), "test3": (X_test3, Y_test3)}.items():
            X, Y = dataset
            rmse = torch.sqrt(torch.mean(torch.square(Y - expr(X))))

            # log the metrics
            mlflow.log_metric(f"{name}/RMSE", rmse.item())


if __name__ == "__main__":

    register_isotherm_config()
    main()
