import torch

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import mlflow

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

def sanity_check_experiment():

    noise_level = 1e-3
    n_data = 20
    n_experiment_runs = 10
    n_opt_steps = 250
    true_expr = "(X * X) + 1"
    X = torch.linspace(-5, 5, steps=n_data)
    Y = torch.square(X) + 1.0 + torch.randn(n_data) * noise_level

    interpreter = TorchEquationInterpreter()

    # load the model from file
    model = torch.load("results/gvae_pretrained.pth")

    for i in range(n_experiment_runs):

        print(f"Started Run #{ i }")

        model.eval()
        expr = model.find_expression_for(X=X, Y=Y, num_opt_steps=n_opt_steps)

        plot_expression(expr, interpreter)
        plt.savefig(f"results/solution_{ i }.pdf")

        with open(f"results/solution.txt") as f:
            f.writelines([
                f"True Expression= {true_expr}",
                f"Solution Found= { ''.join(expr) }",
                f"RMSE= { torch.sqrt(torch.mean(torch.square(interpreter.evaluate(expr, X) - Y))) }"
            ])

        print(f"Ended Run #{ i }")


def isotherm_experiments():
    noise_level = 1e-3
    n_data = 20
    n_experiment_runs = 10
    n_opt_steps = 250
    true_expr = "0.75 * (X * X) + 1.25"
    X = torch.linspace(-5, 5, steps=n_data)
    Y = 0.75 * torch.square(X) + 1.25 + torch.randn(n_data) * noise_level

    # load the model from file
    model = torch.load("results/gvae_pretrained_parametric.pth")

    for i in range(n_experiment_runs):
        print(f"Started Run #{i}")

        model.eval()
        expr = model.find_expression_for(X=X, Y=Y, num_opt_steps=n_opt_steps)

        plot_expression(expr)
        plt.savefig(f"results/solution_{i}.pdf")

        with open(f"results/solution.txt") as f:
            f.writelines([
                f"True Expression= {true_expr}",
                f"Solution Found= {''.join(expr)}",
                f"RMSE= {torch.sqrt(torch.mean(torch.square(interpreter.evaluate(expr, X) - Y)))}"
            ])

        print(f"Ended Run #{i}")


@hydra.main(version_base="1.2", config_path="../config", config_name="symbolic_regression")
def main(cfg: SymbolicIsothermExperimentConfig):

    n_opt_steps = cfg.n_opt_steps

    # setup the symbolic regression problem
    problem: SymbolicIsothermProblem = hydra.utils.instantiate(cfg.problem)

    X, Y = problem.training_data.tensor

    # load the model from file
    model = torch.load(to_absolute_path("results/gvae_pretrained_parametric.pth"))

    mlflow.set_tracking_uri(to_absolute_path("./mlruns"))
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():

        # log the parameters
        mlflow.log_param("isotherm_model", cfg.problem.isotherm_model.name)
        mlflow.log_param("n_opt_steps", cfg.n_opt_steps)

        # fit the expression
        torch.random.seed()
        model.eval()
        expr = model.find_expression_for(X=X, Y=Y, num_opt_steps=n_opt_steps)

        rmse = torch.sqrt(torch.mean(torch.square(Y - expr(X))))
        mlflow.log_metric("training/RMSE", rmse.item())

        # compute the metrics
        for name, dataset in problem.test_data.items():
            X, Y = dataset.tensor
            rmse = torch.sqrt(torch.mean(torch.square(Y - expr(X))))

            # log the metrics
            mlflow.log_metric(f"{name}/RMSE", rmse.item())

        # generate the artifacts
        plot_expression(expr, domain=(0, 250), data=(X, Y))
        plot_filename = f"solution_{ cfg.problem.isotherm_model.name.lower() }"
        plot_extensions = ['pdf', 'png']
        for ext in plot_extensions:
            filename = plot_filename + "." + ext
            plt.savefig(filename)
            mlflow.log_artifact(filename)


if __name__ == "__main__":

    register_isotherm_config()
    main()
