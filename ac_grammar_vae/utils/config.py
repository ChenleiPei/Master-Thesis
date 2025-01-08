import yaml
import datetime
from typing import Any
import os
import omegaconf
import hydra


def fullname(cls: Any) -> str:
    module = cls.__module__
    if module is None or module == str.__class__.__module__:
        return cls.__qualname__
    return ".".join([module, cls.__qualname__])


def load_config_from_run_artifacts(artifacts_path):

    hydra_dir = os.path.join(artifacts_path, '.hydra')
    config_file = os.path.join(hydra_dir, 'config.yaml')
    config = omegaconf.OmegaConf.load(config_file)
    return config