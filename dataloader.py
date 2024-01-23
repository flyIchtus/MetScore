import logging
import os.path

import numpy as np
import pandas as pd
from configurable import Configurable

from abc import ABC, abstractmethod
from typing import Type

from dataset import Dataset, RealDataset, FakeDataset, ObsDataset


class DataLoader(ABC, Configurable):

    def __init__(self):
        self._real_dataset: Type[Dataset] = None
        self._fake_dataset: Type[Dataset] = None
        self._obs_dataset: Type[Dataset] = None
        self._data_length = 0
        self._current_index = 0

    @property
    def real_dataset(self) -> Type[Dataset]:
        return self._real_dataset

    @real_dataset.setter
    def real_dataset(self, dataset: Type[Dataset]):
        if not issubclass(type(dataset), Dataset):
            raise ValueError("real_dataset needs to be a subclass of Dataset")
        self._real_dataset = dataset
    @property
    def fake_dataset(self) -> Type[Dataset]:
        return self._fake_dataset

    @fake_dataset.setter
    def fake_dataset(self, dataset: Type[Dataset]):
        if not issubclass(type(dataset), Dataset):
            raise ValueError("fake_dataset need to be a subclass of Dataset")
        self._fake_dataset = dataset
    @property
    def obs_dataset(self) -> Type[Dataset]:
        return self._obs_dataset

    @obs_dataset.setter
    def obs_dataset(self, dataset: Type[Dataset]):
        if not issubclass(type(dataset), Dataset):
            raise ValueError("obs_dataset needs to be a subclass of Dataset")
        self._obs_dataset = dataset
    @property
    def current_index(self) -> int:
        return self._current_index

    @current_index.setter
    def current_index(self, value: int):
        self._current_index = value

    # @abstractmethod
    # def _collate_fn(self, fake_samples, real_samples, obs_samples):
    #     pass

    def get_all_data(self):
        return self._real_dataset.get_all_data(), self._fake_dataset.get_all_data(), self._obs_dataset.get_all_data()

    def __iter__(self):
        self.current_index = 0
        return self

    @abstractmethod
    def __next__(self):
        pass

    def __len__(self):
        return self._data_length


class DateDataloader(DataLoader):

    def __init__(self, config_data, use_cache=False):
        # Appel du __init__ de la classe mère
        super().__init__()

        # add df0, LT, dh to config_data, start_time to config_data dataset
        # augment_dict = {'df0': self.df0, 'LT': config_data['Lead_Times'], 'dh': config_data['dh'], 'start_time': config_data['start_time']}
        config_data['real_dataset_config'].update(config_data)
        config_data['fake_dataset_config'].update(config_data)
        config_data['obs_dataset_config'].update(config_data)


        # Instanciation des datasets dans DateDataloader
        self.real_dataset = RealDataset.fromConfig(config_data['real_dataset_config'], use_cache=use_cache)
        self.fake_dataset = FakeDataset.fromConfig(config_data['fake_dataset_config'], use_cache=use_cache)
        self.obs_dataset = ObsDataset.fromConfig(config_data['obs_dataset_config'], use_cache=use_cache)
        self._data_length = min(len(self.real_dataset), len(self.fake_dataset), len(self.obs_dataset))

    def __next__(self):
        if self.current_index < self._data_length:
            fake_samples = np.array([self.fake_dataset[ self.current_index + i] for i in range(self.batch_size)])
            obs_samples = np.array([self.obs_dataset[self.current_index + i] for i in range(self.batch_size)])
            real_samples = np.array([self.real_dataset[self.current_index + i] for i in range(self.batch_size)])
            self.current_index += min(self.batch_size, self._data_length - self.current_index)

            return fake_samples[0], real_samples[0], obs_samples
        else:
            raise StopIteration

    def get_all_data(self):
        real = self._real_dataset.get_all_data()
        fake = self._fake_dataset.get_all_data()
        obs = self._obs_dataset.get_all_data()
        real, fake = self.randomize_and_cut(real, fake)
        return real, fake, obs

    def randomize_and_cut(self, data1, data2):
        data1shuf = np.random.permutation(data1)
        data2shuf = np.random.permutation(data2)
        cut = min([self.maxNsamples,data1shuf.shape[0],data2shuf.shape[0]])
        if cut<self.maxNsamples:
            logging.warning(f"maxNsamples set to {self.maxNsamples} but not enough samples ({cut}). Continuing with {cut} samples.")
        return data1shuf[:cut], data2shuf[:cut]
