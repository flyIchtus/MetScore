import numpy as np

from configurable import Configurable
from dataset import ObsDataset, FakeDataset, RealDataset


class DataLoader(Configurable):
    def __init__(self, config_data, use_cache=False):
        """
        Basic dataloader class for loading data from datasets
        config : dict
        """
        super().__init__()
        self.real_dataset = RealDataset.fromConfig(config_data['real_dataset_config'], use_cache=use_cache)
        self.fake_dataset = FakeDataset.fromConfig(config_data['fake_dataset_config'], use_cache=use_cache)
        self.obs_dataset = ObsDataset.fromConfig(config_data['obs_dataset_config'], use_cache=use_cache)

        self.data_length = min(len(self.real_dataset), len(self.fake_dataset), len(self.obs_dataset))
        self.batch_size = config_data['batch_size']
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index < self.data_length:
            fake_samples, real_samples, obs_samples = [], [], []
            for i in range(min(self.batch_size, self.data_length - self.current_index)):
                fake_samples.append(self.fake_dataset[self.current_index + i])
                obs_samples.append(self.obs_dataset[self.current_index + i])
                real_samples.append(self.real_dataset[self.current_index + i])

            self.current_index += self.batch_size

            return self._collate_fn(fake_samples, real_samples, obs_samples)
        else:
            raise StopIteration

    def __len__(self):
        return self.data_length

    def _collate_fn(self, fake_samples, real_samples, obs_samples):
        return np.array(fake_samples),  np.array(real_samples),  np.array(obs_samples)

    def get_all_data(self):
        return (
            self.real_dataset.get_all_data(),
            self.fake_dataset.get_all_data(),
            self.obs_dataset.get_all_data()
        )


class DateDataloader(DataLoader):

    def __next__(self):
        if self.current_index < self.data_length:
            fake_samples = [self.fake_dataset[self.current_index + i] for i in range(self.batch_size)]
            fake_dates = [sample['date'] for sample in fake_samples]
            real_samples, obs_samples = [], []
            for date in fake_dates:
                real_samples.append(self.real_dataset[date])
                obs_samples.append(self.obs_dataset[date])

            self.current_index += self.batch_size

            yield fake_samples, real_samples, obs_samples
        else:
            raise StopIteration
