import abc
from abc import ABC
import torch

from ac_grammar_vae.utils.random_seed import set_random_seed


class IsothermModel(ABC):

    @classmethod
    @abc.abstractmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        ...

    @abc.abstractmethod
    def __call__(self, c):
        ...

    @staticmethod
    @abc.abstractmethod
    def num_parameters():
        ...

    @staticmethod
    @abc.abstractmethod
    def name():
        ...

    @staticmethod
    @abc.abstractmethod
    def latex_equation():
        ...


class Langmuir(IsothermModel):

    def __init__(self, s_T, k):
        self.s_T = s_T
        self.k = k

    def __call__(self, c):
        kc = self.k * c
        return self.s_T * kc / (1 + kc)

    @staticmethod
    def num_parameters():
        return 2

    @staticmethod
    def name():
        return "Langmuir"

    @staticmethod
    def latex_equation():
        return "$s_T\\frac{kc}{1 + kc}$"

    @classmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        set_random_seed(random_seed)

        if as_numpy:
            return Langmuir(
                s_T= torch.distributions.Exponential(rate=0.015).sample().numpy(),
                k= torch.distributions.Exponential(4).sample().numpy()
            )
        else:
            return Langmuir(
                s_T=torch.distributions.Exponential(rate=0.015).sample(),
                k=torch.distributions.Exponential(4).sample()
            )


class TwoSiteLangmuir(IsothermModel):

    def __init__(self, s_T, f_1, f_2, k_1, k_2):
        self.s_T = s_T
        self.f_1 = f_1
        self.f_2 = f_2
        self.k_1 = k_1
        self.k_2 = k_2

    def __call__(self, c):
        k_1c = self.k_1 * c
        k_2c = self.k_2 * c
        return self.s_T * (self.f_1 * k_1c / (1 + k_1c) + self.f_2 * k_2c / (1 + k_2c))

    @staticmethod
    def num_parameters():
        return 5

    @staticmethod
    def name():
        return "Two Site Langmuir"

    @staticmethod
    def latex_equation():
        return "$s_T \\left( \\frac{f_1 k_1 c}{1 + k_1 c} + \\frac{f_2 k_2 c}{1 + k_2 c} \\right)$"

    @classmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        set_random_seed(random_seed)

        if as_numpy:
            return TwoSiteLangmuir(
                s_T=torch.distributions.Exponential(rate=0.015).sample().numpy(),
                k_1=torch.distributions.Exponential(8).sample().numpy(),
                k_2=torch.distributions.Exponential(8).sample().numpy(),
                f_1=torch.distributions.Exponential(4).sample().numpy(),
                f_2=torch.distributions.Exponential(4).sample().numpy()
            )
        else:
            return TwoSiteLangmuir(
                s_T=torch.distributions.Exponential(rate=0.015).sample(),
                k_1=torch.distributions.Exponential(8).sample(),
                k_2=torch.distributions.Exponential(8).sample(),
                f_1=torch.distributions.Exponential(2).sample(),
                f_2=torch.distributions.Exponential(2).sample()
            )


class ModifiedLangmuir(IsothermModel):

    def __init__(self, s_T, k_1, k_2):
        self.s_T = s_T
        self.k_1 = k_1
        self.k_2 = k_2

    def __call__(self, c):
        k_1c = self.k_1 * c
        k_2c = self.k_2 * c
        return self.s_T * k_1c / (1 + k_1c) / (1 + k_2c)

    @staticmethod
    def num_parameters():
        return 3

    @staticmethod
    def name():
        return "Modified Langmuir"

    @staticmethod
    def latex_equation():
        return "$s_T\\frac{k_1 c}{1 + k_1 c} \\frac{1}{1 + k_2 c}$"

    @classmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        set_random_seed(random_seed)

        if as_numpy:
            return ModifiedLangmuir(
                s_T=torch.distributions.Exponential(rate=0.015).sample().numpy(),
                k_1=torch.distributions.Exponential(4).sample().numpy(),
                k_2=torch.distributions.Exponential(100).sample().numpy()
            )
        else:
            return ModifiedLangmuir(
                s_T=torch.distributions.Exponential(rate=0.015).sample(),
                k_1=torch.distributions.Exponential(4).sample(),
                k_2=torch.distributions.Exponential(100).sample()
            )


class BrunauerEmmettTeller(IsothermModel):

    def __init__(self, k_1, k_2, k_3):
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3

    def __call__(self, c):
        return (self.k_1 * c) / (1 + self.k_2 * c) / (1 - self.k_3 * c)

    @staticmethod
    def num_parameters():
        return 3

    @staticmethod
    def name():
        return "Brunauer Emmett Teller"

    @staticmethod
    def latex_equation():
        return "$\\frac{k_1 c}{1 + k_2 c} \\frac{1}{1 - k_3 c}$"

    @classmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        set_random_seed(random_seed)

        if as_numpy:
            return BrunauerEmmettTeller(
                k_1=torch.distributions.Exponential(0.25).sample().numpy(),
                k_2=torch.distributions.Exponential(4).sample().numpy(),
                k_3=torch.distributions.Exponential(100).sample().numpy()
            )
        else:
            return BrunauerEmmettTeller(
                k_1=torch.distributions.Exponential(0.25).sample(),
                k_2=torch.distributions.Exponential(8).sample(),
                k_3=torch.distributions.Exponential(150).sample()
            )


class FarleyDzombakMorel(IsothermModel):

    def __init__(self, s_T, k_1, k_2, k_3, X, X_c):
        self.s_T = s_T
        self.k_1 = k_1
        self.k_2 = k_2
        self.k_3 = k_3
        self.X = X
        self.X_c = X_c

    def __call__(self, c):
        k_1c = self.k_1 * c
        k_2c = self.k_2 * c

        result = self.s_T * k_1c / ( 1 + k_1c)
        result += ((self.X - self.s_T) / (1 + k_1c) + (self.k_1 * self.X_c) / (1 + k_1c)) * k_2c / (1 - k_2c)
        result -= self.k_2 / self.k_3 * c
        return result

    @staticmethod
    def num_parameters():
        return 6

    @staticmethod
    def latex_equation():
        return "$\\ldots$"

    @staticmethod
    def name():
        return "Farley Dzombak Morel"

    @classmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        set_random_seed(random_seed)

        if as_numpy:
            return FarleyDzombakMorel(
                s_T=torch.distributions.Exponential(rate=0.015).sample().numpy(),
                k_1=torch.distributions.Exponential(4).sample().numpy(),
                k_2=torch.distributions.Exponential(100).sample().numpy(),
                k_3=torch.distributions.Exponential(4).sample().numpy(),
                X=torch.distributions.Exponential(rate=0.03).sample().numpy(),
                X_c=torch.distributions.Exponential(rate=0.03).sample().numpy()
            )
        else:
            return FarleyDzombakMorel(
                s_T=torch.distributions.Exponential(rate=0.015).sample(),
                k_1=torch.distributions.Exponential(4).sample(),
                k_2=torch.distributions.Exponential(100).sample(),
                k_3=torch.distributions.Exponential(4).sample(),
                X=torch.distributions.Exponential(rate=0.03).sample(),
                X_c=torch.distributions.Exponential(rate=0.03).sample()
            )


class Freundlich(IsothermModel):

    def __init__(self, K_F, alpha):
        self.K_F = K_F
        self.alpha = alpha

    def __call__(self, c):
        return (c ** self.alpha) * self.K_F

    @staticmethod
    def num_parameters():
        return 2

    @staticmethod
    def name():
        return "Freundlich"

    @staticmethod
    def latex_equation():
        return "$K_F c^\\alpha$"

    @classmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        set_random_seed(random_seed)

        if as_numpy:
            return Freundlich(
                K_F=torch.distributions.Exponential(0.05).sample().numpy(),
                alpha=torch.distributions.Exponential(4).sample().numpy()
            )
        else:
            return Freundlich(
                K_F=torch.distributions.Exponential(0.05).sample(),
                alpha=torch.distributions.Exponential(4).sample()
            )


class GeneralLangmuirFreundlich(IsothermModel):

    def __init__(self, s_T, k, alpha):
        self.s_T = s_T
        self.k = k
        self.alpha = alpha

    def __call__(self, c):
        kc = (self.k * c) ** self.alpha
        return self.s_T * kc / (1 + kc)

    @staticmethod
    def num_parameters():
        return 3

    @staticmethod
    def name():
        return "General Langmuir Freundlich"

    @staticmethod
    def latex_equation():
        return "$s_T\\frac{(kc)^\\alpha}{1 + (kc)^\\alpha}$"

    @classmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        set_random_seed(random_seed)

        if as_numpy:
            return GeneralLangmuirFreundlich(
                s_T=torch.distributions.Exponential(rate=0.015).sample().numpy(),
                k=torch.distributions.Exponential(4).sample().numpy(),
                alpha=torch.distributions.Exponential(4).sample().numpy()
            )
        else:
            return GeneralLangmuirFreundlich(
                s_T=torch.distributions.Exponential(rate=0.015).sample(),
                k=torch.distributions.Exponential(4).sample(),
                alpha=torch.distributions.Exponential(4).sample()
            )


class GeneralFreundlich(IsothermModel):

    def __init__(self, s_T, k, alpha):
        self.s_T = s_T
        self.k = k
        self.alpha = alpha

    def __call__(self, c):
        kc = (self.k * c)
        return self.s_T * (kc / (1 + kc)) ** self.alpha

    @staticmethod
    def num_parameters():
        return 3

    @staticmethod
    def name():
        return "General Freundlich"

    @staticmethod
    def latex_equation():
        return "$s_T\\left(\\frac{kc}{1 + kc}\\right)^\\alpha$"

    @classmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        set_random_seed(random_seed)

        if as_numpy:
            return GeneralFreundlich(
                s_T=torch.distributions.Exponential(rate=0.015).sample().numpy(),
                k=torch.distributions.Exponential(4).sample().numpy(),
                alpha=torch.distributions.Exponential(4).sample().numpy()
            )
        else:
            return GeneralFreundlich(
                s_T=torch.distributions.Exponential(rate=0.015).sample(),
                k=torch.distributions.Exponential(4).sample(),
                alpha=torch.distributions.Exponential(4).sample()
            )


class RedlichPeterson(IsothermModel):

    def __init__(self, s_T, k, alpha):
        self.s_T = s_T
        self.k = k
        self.alpha = alpha

    def __call__(self, c):
        kc = (self.k * c)
        return self.s_T * kc / (1 + kc ** self.alpha)

    @staticmethod
    def num_parameters():
        return 3

    @staticmethod
    def name():
        return "Redlich Peterson"

    @staticmethod
    def latex_equation():
        return "$s_T\\frac{kc}{1 + (kc)^\\alpha}$"

    @classmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        set_random_seed(random_seed)

        if as_numpy:
            return RedlichPeterson(
                s_T=torch.distributions.Exponential(rate=0.015).sample().numpy(),
                k=torch.distributions.Exponential(4).sample().numpy(),
                alpha=torch.distributions.Exponential(0.75).sample().numpy()
            )
        else:
            return RedlichPeterson(
                s_T=torch.distributions.Exponential(rate=0.015).sample(),
                k=torch.distributions.Exponential(4).sample(),
                alpha=torch.distributions.Exponential(0.75).sample()
            )


class Toth(IsothermModel):

    def __init__(self, s_T, k, alpha):
        self.s_T = s_T
        self.k = k
        self.alpha = alpha

    def __call__(self, c):
        kc = (self.k * c)
        return self.s_T * kc / (1 + kc ** self.alpha) ** (1 / self.alpha)

    @staticmethod
    def num_parameters():
        return 3

    @staticmethod
    def name():
        return "Toth"

    @staticmethod
    def latex_equation():
        return "$s_T\\frac{kc}{(1 + (kc)^\\alpha)^{1/\\alpha}}$"

    @classmethod
    def from_parameter_prior_sample(cls, as_numpy=False, random_seed=None, **kwargs):
        set_random_seed(random_seed)

        if as_numpy:
            return Toth(
                s_T=torch.distributions.Exponential(rate=0.015).sample().numpy(),
                k=torch.distributions.Exponential(4).sample().numpy(),
                alpha=torch.distributions.Exponential(0.75).sample().numpy()
            )
        else:
            return Toth(
                s_T=torch.distributions.Exponential(rate=0.015).sample(),
                k=torch.distributions.Exponential(4).sample(),
                alpha=torch.distributions.Exponential(0.75).sample()
            )


all_isotherms = list(IsothermModel.__subclasses__())


def find_isotherm_by_name(name):
    return next(iter(filter(lambda i: i.name() == name, all_isotherms)))