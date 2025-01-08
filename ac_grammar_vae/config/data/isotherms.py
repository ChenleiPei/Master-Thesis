from dataclasses import dataclass
from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from ac_grammar_vae.utils.config import fullname
from ac_grammar_vae.data.sorption.isotherm import Langmuir, ModifiedLangmuir, TwoSiteLangmuir, Freundlich, GeneralFreundlich, GeneralLangmuirFreundlich, Toth, RedlichPeterson, FarleyDzombakMorel, BrunauerEmmettTeller


@dataclass
class IsothermModelConfig:
    _target_: str = MISSING
    name: str = MISSING
    latex_equation: str = MISSING
    random_seed: int = 20220


@dataclass
class LangmuirModelConfig(IsothermModelConfig):
    _target_: str = fullname(Langmuir.from_parameter_prior_sample)
    name: str = Langmuir.name()
    latex_equation: str = Langmuir.latex_equation()


@dataclass
class ModifiedLangmuirModelConfig(IsothermModelConfig):
    _target_: str = fullname(ModifiedLangmuir.from_parameter_prior_sample)
    name: str = ModifiedLangmuir.name()
    latex_equation: str = ModifiedLangmuir.latex_equation()


@dataclass
class TwoSiteLangmuirModelConfig(IsothermModelConfig):
    _target_: str = fullname(TwoSiteLangmuir.from_parameter_prior_sample)
    name: str = TwoSiteLangmuir.name()
    latex_equation: str = TwoSiteLangmuir.latex_equation()


@dataclass
class FreundlichModelConfig(IsothermModelConfig):
    _target_: str = fullname(Freundlich.from_parameter_prior_sample)
    name: str = Freundlich.name()
    latex_equation: str = Freundlich.latex_equation()


@dataclass
class GeneralFreundlichModelConfig(IsothermModelConfig):
    _target_: str = fullname(GeneralFreundlich.from_parameter_prior_sample)
    name: str = GeneralFreundlich.name()
    latex_equation: str = GeneralFreundlich.latex_equation()


@dataclass
class GeneralLangmuirFreundlichModelConfig(IsothermModelConfig):
    _target_: str = fullname(GeneralLangmuirFreundlich.from_parameter_prior_sample)
    name: str = GeneralLangmuirFreundlich.name()
    latex_equation: str = GeneralLangmuirFreundlich.latex_equation()


@dataclass
class TothModelConfig(IsothermModelConfig):
    _target_: str = fullname(Toth.from_parameter_prior_sample)
    name: str = Toth.name()
    latex_equation: str = Toth.latex_equation()


@dataclass
class RedlichPetersonModelConfig(IsothermModelConfig):
    _target_: str = fullname(RedlichPeterson.from_parameter_prior_sample)
    name: str = RedlichPeterson.name()
    latex_equation: str = RedlichPeterson.latex_equation()


@dataclass
class FarleyDzombakMorelModelConfig(IsothermModelConfig):
    _target_: str = fullname(FarleyDzombakMorel.from_parameter_prior_sample)
    name: str = FarleyDzombakMorel.name()
    latex_equation: str = FarleyDzombakMorel.latex_equation()


@dataclass
class BrunauerEmmettTellerModelConfig(IsothermModelConfig):
    _target_: str = fullname(BrunauerEmmettTeller.from_parameter_prior_sample)
    name: str = BrunauerEmmettTeller.name()
    latex_equation: str = BrunauerEmmettTeller.latex_equation()


def register_isotherms():
    cs = ConfigStore()

    cs.store(name='langmuir', group='problem/isotherm_model', node=LangmuirModelConfig)
    cs.store(name='modified-langmuir', group='problem/isotherm_model', node=ModifiedLangmuirModelConfig)
    cs.store(name='twosite-langmuir', group='problem/isotherm_model', node=TwoSiteLangmuirModelConfig)
    cs.store(name='general-langmuir-freundlich', group='problem/isotherm_model', node=GeneralLangmuirFreundlichModelConfig)
    cs.store(name='freundlich', group='problem/isotherm_model', node=FreundlichModelConfig)
    cs.store(name='general-freundlich', group='problem/isotherm_model', node=GeneralFreundlichModelConfig)
    cs.store(name='toth', group='problem/isotherm_model', node=TothModelConfig)
    cs.store(name='redlich-peterson', group='problem/isotherm_model', node=RedlichPetersonModelConfig)
    cs.store(name='farley-dzombak-morel', group='problem/isotherm_model', node=FarleyDzombakMorelModelConfig)
    cs.store(name='brunauer-emmett-teller', group='problem/isotherm_model', node=BrunauerEmmettTellerModelConfig)