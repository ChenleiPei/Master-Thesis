import abc
from abc import ABC
from typing import Dict
from ac_grammar_vae.data.sorption.dataset import BaseSorptionDataset, SyntheticSorptionDataset


class SymbolicIsothermProblem(ABC):

    @property
    @abc.abstractmethod
    def training_data(self) -> BaseSorptionDataset:
        ...

    @property
    @abc.abstractmethod
    def validation_data(self) -> BaseSorptionDataset:
        ...

    @property
    @abc.abstractmethod
    def test_data(self) -> Dict[str, BaseSorptionDataset]:
        ...


class SyntheticSymbolicIsothermProblem(SymbolicIsothermProblem):

    def __init__(self, isotherm_model, training_data, validation_data, test_data):

        self.isotherm_model = isotherm_model

        if isinstance(training_data, SyntheticSorptionDataset):
            self._training_data = training_data
        else:
            self._training_data = training_data(isotherm_model= isotherm_model)

        if isinstance(validation_data, SyntheticSorptionDataset):
            self._validation_data = validation_data
        else:
            self._validation_data = validation_data(isotherm_model= isotherm_model)

        self._test_data = {}
        for name, ds in test_data.items():
            if isinstance(ds, SyntheticSorptionDataset):
                self._test_data[name] = ds
            else:
                self._test_data[name] = ds(isotherm_model=isotherm_model)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.isotherm_model.name()} (synthetic data problem)"

    @property
    def training_data(self) -> BaseSorptionDataset:
        return self._training_data

    @property
    def validation_data(self) -> BaseSorptionDataset:
        return self._validation_data

    @property
    def test_data(self) -> Dict[str, BaseSorptionDataset]:
        return self._test_data