from dataclasses import dataclass, field
from typing import Dict, List, Any, Union
from omegaconf import MISSING

import hydra
from hydra.conf import HydraConf
from hydra.core.config_store import ConfigStore

from .data.base_synthetic_isotherm import IsothermDatasetConfig
from .data.problem import IsothermProblemConfig
from .data.isotherms import register_isotherms
"""
from .solver.base_solver import SymbolicSolverConfig
from .solver.prte_solver_config import PRTESolverConfig
from .solver.bsr_solver_config import BSRSolverConfig
from .solver.knowledge_config import HinzSoprtionKnowledgeConfig
from .launcher.launcher_config import LocalConfig, SlurmConfig
from .evaluation.evaluation_hook_config import EvaluationHookConfig, MLFlowEvaluationHookConfig, PlotIsothermEvaluationHookConfig
from .mlflow import MLFlowConfig
"""


defaults: List[Union[str, Dict[str, str]]] = [
    "_self_",
    {"problem.isotherm_model": "general-langmuir-freundlich"},
    {"solver": "prte"},
    {'override hydra/launcher' : 'submitit_slurm'}
]


@dataclass
class SymbolicIsothermExperimentConfig:

    # general config
    defaults: List[Any] = field(default_factory=lambda: defaults)
    hydra: HydraConf = HydraConf()

    # Experiment Configuration
    problem: IsothermProblemConfig = MISSING

    # launcher
    #launcher = SlurmConfig()

    n_opt_steps: int = MISSING
    experiment_name: str = MISSING

def register_isotherm_config():

    register_isotherms()

    cs = ConfigStore()
    cs.store(name='synthetic_isotherm_problem', group='problem', node=IsothermProblemConfig)
