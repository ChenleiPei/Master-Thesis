import torch
import time
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import mlflow

from matplotlib import pyplot as plt
from ac_grammar_vae.model.gvae.interpreter import TorchEquationInterpreter, ExpressionWithParameters

from ac_grammar_vae.config.experiment import register_isotherm_config, SymbolicIsothermExperimentConfig
from ac_grammar_vae.data.sorption.problem import SymbolicIsothermProblem


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
        mlflow.log_param("isotherm", cfg.problem.isotherm_model.name)

        # fit the expression
        torch.random.seed()
        model.eval()
        expr = model.find_expression_for(X=X, Y=Y, num_opt_steps=n_opt_steps)

        mlflow.log_param("complexity", expr.complexity)
        logging.info(f"Complexity: {expr.complexity}")
        mlflow.log_param("structure_complexity", expr.structure_complexity)
        logging.info(f"Structure Complexity: {expr.structure_complexity}")
        mlflow.log_param("Parameter Complexity", expr.parameter_complexity)
        logging.info(f"Parameter Complexity: {expr.parameter_complexity}")

        rmse = torch.sqrt(torch.mean(torch.square(Y - expr(X))))
        mlflow.log_metric("training/RMSE", rmse.item())

        # compute the metrics
        for name, dataset in problem.test_data.items():
            X, Y = dataset.tensor
            rmse = torch.sqrt(torch.mean(torch.square(Y - expr(X))))

            # log the metrics
            mlflow.log_metric(f"{name}/RMSE", rmse.item())


if __name__ == "__main__":
    start_time = time.time()
    register_isotherm_config()
    main()
    print(time.time() - start_time)
