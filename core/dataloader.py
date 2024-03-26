import logging
from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from core.configurable import Configurable
from core.dataset import Dataset, RealDataset, ObsDataset, RandomDataset


# region Base DataLoader

class DataLoader(ABC, Configurable):
    """
    Abstract base class for data loaders.

    This class provides an interface for loading and iterating over datasets.
    It includes properties for managing real, fake, and observed datasets, as well as the current index and data length.

    Attributes:
        _real_dataset (Type[Dataset]): The real dataset.
        _fake_dataset (Type[Dataset]): The fake dataset.
        _obs_dataset (Type[Dataset]): The observed dataset.
        _data_length (int): The length of the data.
        _current_index (int): The current index in the data.

    To add a custom DataLoader, follow these steps:

    1. Create a new class that inherits from the `DataLoader` abstract base class (ABC).

    2. Implement the `__next__` method in your custom DataLoader class. This method should define the behavior for
    returning the next sample from the real, fake, and observed datasets. Make sure to handle the case when there are
    no more samples to iterate over by raising a `StopIteration` exception.

    3. (Optional) If your custom DataLoader requires specific configuration options, add them as class attributes or
    properties, and make sure to update the `required_keys` class attribute accordingly. You can also add custom
    methods to your DataLoader class if needed.

    4. To use your custom DataLoader, create an instance of it with the appropriate configuration and pass it to the
    experiment or other components that require a DataLoader.

    Example:

    Here's an example of a custom DataLoader called `MyCustomDataLoader`:

    ```python
    class MyCustomDataLoader(DataLoader):
        required_keys = ['my_custom_key']

        def __init__(self, config_data, **kwargs):
            super().__init__(config_data, **kwargs)
            real_samples = np.random.rand(100, 3)
            fake_samples = np.random.rand(100, 3)
            obs_samples = np.random.rand(100, 3)
            data_length = 100
            # self.my_custom_key = config_data['my_custom_key']
            # my_custom_key is automatically set as an attribute on the instance by the Configurable base class

        def __next__(self):
            if self.current_index < self._data_length:
                # Your custom logic for returning the next sample from the datasets
                real_sample = self._real_dataset[self.current_index]
                fake_sample = self._fake_dataset[self.current_index]
                obs_sample = self._obs_dataset[self.current_index]
                self.current_index += 1
                return real_sample, fake_sample, obs_sample
            else:
                raise StopIteration
    ```
    """

    def __init__(self, **kwargs):
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
        """
        Return all data from the real, fake, and observed datasets.

        Returns:
            tuple: A tuple containing all data from the real, fake, and observed datasets, respectively.
        """
        return self._real_dataset.get_all_data(), self._fake_dataset.get_all_data(), self._obs_dataset.get_all_data()

    def __iter__(self):
        """
        Implement the iterator protocol, allowing the DataLoader instance to be iterated over.

        Returns:
            DataLoader: The DataLoader instance as an iterator.
        """
        self.current_index = 0
        return self

    @abstractmethod
    def __next__(self):
        """
        Abstract method for implementing the iterator protocol.

        This method should be implemented in subclasses to provide the specific behavior for returning the next sample from the datasets.

        Raises:
            StopIteration: When there are no more samples to iterate over.
        """
        pass

    def __len__(self):
        """
        Return the length of the data.

        Returns:
            int: The length of the data, which is the minimum length among the real, fake, and observed datasets.
        """
        return self._data_length


# endregion

# region Custom Dataloader

class DateDataloader(DataLoader):
    """
       A data loader for date-based datasets.

       This class extends the DataLoader base class and loads real, fake, and observed datasets based on the provided configuration.
       It iterates over the datasets and returns samples for each iteration.

       Attributes:
           real_dataset (Type[RealDataset]): The real dataset.
           fake_dataset (Type[Dataset]): The fake dataset.
           obs_dataset (Type[ObsDataset]): The observed dataset.
           _data_length (int): The length of the data.
           _current_index (int): The current index in the data.
    """

    def __init__(self, config_data, use_cache=False, **kwargs):
        # Appel du __init__ de la classe mère
        super().__init__()
        del config_data['type']  # ensuring 'type' of dataloader does not leak on 'type' of datasets
        config_data['real_dataset_config'].update(config_data)
        config_data['fake_dataset_config'].update(config_data)
        config_data['obs_dataset_config'].update(config_data)

        self.real_dataset = RealDataset.fromConfig(config_data['real_dataset_config'], use_cache=use_cache)
        self.fake_dataset = Dataset.from_typed_config(config_data['fake_dataset_config'], use_cache=use_cache)
        self.obs_dataset = ObsDataset.fromConfig(config_data['obs_dataset_config'], use_cache=use_cache)
        self._data_length = min(len(self.real_dataset), len(self.fake_dataset), len(self.obs_dataset))
        logging.debug(f"Dataset length is {self._data_length}")

    def __next__(self):
        if self.current_index < self._data_length:
            fake_samples = np.array([self.fake_dataset[self.current_index + i] for i in range(self.batch_size)])
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
        cut = min([self.maxNsamples, data1shuf.shape[0], data2shuf.shape[0]])
        if cut < self.maxNsamples:
            logging.warning(
                f"maxNsamples set to {self.maxNsamples} but not enough samples ({cut}). Continuing with {cut} samples.")
        return data1shuf[:cut], data2shuf[:cut]


class RandomDataloader(DataLoader):
    """
       A data loader for random datasets.

       This class extends the DataLoader base class and loads random real and fake datasets based on the provided configuration.
       It iterates over the datasets and returns samples for each iteration.

       Attributes:
           real_dataset (Type[RandomDataset]): The real dataset.
           fake_dataset (Type[RandomDataset]): The fake dataset.
           _data_length (int): The length of the data.
           _current_index (int): The current index in the data.
    """

    required_keys = ['real_dataset_config', 'fake_dataset_config']

    def __init__(self, config_data, use_cache=False, **kwargs):
        # Appel du __init__ de la classe mère
        super().__init__()

        config_data['real_dataset_config'].update(config_data)
        config_data['fake_dataset_config'].update(config_data)
        self.real_dataset = RandomDataset.fromConfig(config_data['real_dataset_config'], use_cache=use_cache)
        self.fake_dataset = RandomDataset.fromConfig(config_data['fake_dataset_config'], use_cache=use_cache)
        self._data_length = min(len(self.real_dataset), len(self.fake_dataset))
        logging.debug(f"Dataset length is {self._data_length}")

    def __next__(self):
        if self.current_index < self._data_length:
            fake_samples = np.array(
                [self.fake_dataset[self.current_index + i] for i in range(self.fake_dataset.batch_size)])
            real_samples = np.array(
                [self.real_dataset[self.current_index + i] for i in range(self.real_dataset.batch_size)])
            self.current_index += min(self.batch_size, self._data_length - self.current_index)
            return fake_samples[0], real_samples[0], None
        else:
            raise StopIteration

    def get_all_data(self):
        real = self._real_dataset.get_all_data()
        fake = self._fake_dataset.get_all_data()
        real, fake = self.randomize_and_cut(real, fake)
        return real, fake, None

    def randomize_and_cut(self, data1, data2):
        data1shuf = np.random.permutation(data1)
        data2shuf = np.random.permutation(data2)
        cut = min([self.maxNsamples, data1shuf.shape[0], data2shuf.shape[0]])
        if cut < self.maxNsamples:
            logging.warning(
                f"maxNsamples set to {self.maxNsamples} but not enough samples ({cut}). Continuing with {cut} samples.")
        return data1shuf[:cut], data2shuf[:cut]

# endregion
