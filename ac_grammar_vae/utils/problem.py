from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Dict, Any

import hydra
from hydra.core.config_store import ConfigStore

from .base_synthetic_isotherm import IsothermDatasetConfig
from .isotherms import *

from ibsr.utils.config import fullname
from ibsr.data.sorption.isotherm import IsothermModel
from ibsr.data.sorption.problem import SyntheticSymbolicIsothermProblem


@dataclass
class IsothermProblemConfig:

    _target_:str = fullname(SyntheticSymbolicIsothermProblem)

    isotherm_model : IsothermModelConfig = MISSING

    # configure the datasets
    training_data: IsothermDatasetConfig = IsothermDatasetConfig(n_samples=20, random_seed=20231, noise_level=0.1)
    validation_data: IsothermDatasetConfig = IsothermDatasetConfig(n_samples=20, random_seed=20222, noise_level=0.1)
    test_data: Dict[str, IsothermDatasetConfig] = field(default_factory=lambda: {
        'test1' : IsothermDatasetConfig(n_samples=20, random_seed=20223, noise_level=0.0, c_limits=(20, 100)),
        'test2' : IsothermDatasetConfig(n_samples=20, random_seed=20224, noise_level=0.0, c_limits=(0, 20)),
        'test3' : IsothermDatasetConfig(n_samples=20, random_seed=20225, noise_level=0.0, c_limits=(100, 150))
    })

