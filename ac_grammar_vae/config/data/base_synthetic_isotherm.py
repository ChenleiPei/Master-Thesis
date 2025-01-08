from dataclasses import dataclass
from omegaconf import MISSING
from typing import Tuple

from .isotherms import IsothermModelConfig

from ac_grammar_vae.utils.config import fullname
from ac_grammar_vae.data.sorption.dataset import SyntheticSorptionDataset


@dataclass
class IsothermDatasetConfig:

    _partial_: bool = True
    _target_:str = fullname(SyntheticSorptionDataset)

    n_samples : int = MISSING
    random_seed: int = MISSING
    noise_level : float = 0.1
    c_limits: Tuple[float, float] = (20.0, 100.0)

