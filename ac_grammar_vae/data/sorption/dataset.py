import torch
from torch.utils.data import Dataset
import abc


class BaseSorptionDataset(Dataset):

    @abc.abstractmethod
    def __len__(self):
        ...

    @abc.abstractmethod
    def __getitem__(self, item):
        ...

    @abc.abstractproperty
    def tensor(self):
        ...


class SyntheticSorptionDataset(BaseSorptionDataset):

    def __init__(self, n_samples, isotherm_model, noise_level = 0.01, c_limits=(0, 100), random_seed=2022):
        self.n_samples = n_samples
        self.isotherm_model = isotherm_model
        self.noise_level = noise_level
        self.c_limits = c_limits

        torch.random.manual_seed(random_seed)
        input_samples = self.c_limits[0] + torch.rand(self.n_samples) * (self.c_limits[1] - self.c_limits[0])
        self._data = SyntheticSorptionDataset.generate_data(input_samples, self.isotherm_model, self.noise_level)

    @staticmethod
    def generate_data(input_samples, model, noise_level):
        out = model(input_samples)
        noise = torch.randn(*out.shape) * noise_level
        return [input_samples, out + noise]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        x, y = self._data
        return x[item], y[item]

    @property
    def tensor(self):
        return self._data