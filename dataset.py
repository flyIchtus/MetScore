import os
import threading

import numpy as np

from configurable import Configurable
from useful_funcs import obs_clean
from abc import ABC, abstractmethod

from transforms.preprocessor import Preprocessor


def thread_safe_semaphore(func):
    def wrapper(self, file_path):
        with self.load_data_semaphore:
            return func(self, file_path)

    return wrapper


class MemoryCache:
    def __init__(self, use_cache):
        self.cache = {}
        self.use_cache = use_cache

    def add_to_cache(self, key, data):
        if self.use_cache:
            self.cache[key] = data

    def is_cached(self, key):
        if not self.use_cache:
            return False
        return key in self.cache

    def get_from_cache(self, key):
        if not self.use_cache:
            return None
        return self.cache[key]


# TODO: voir si c'est pas mieux d'utiliser la classe de torch
class Dataset(Configurable):
    def __init__(self, config_data, use_cache=True):
        """
        Sample for config yml file:
        data_folder: path to data folder
        batch_size: 32
        preprocessor_config:
            name: Preprocessor
            args:
                arg1: value1
                arg2: value2
        """
        super().__init__()
        self.preprocessor = Preprocessor.fromConfig(config_data.get('preprocessor_config', {'Preprocessor'}))
        self.cache = MemoryCache(use_cache)
        self.file_list = os.listdir(config_data['data_folder'])
        self.load_data_semaphore = threading.Semaphore()

    @abstractmethod
    def _get_filename(self, items):
        pass

    @abstractmethod
    def _load_file(self, file_path):
        pass

    def _load_and_preprocess(self, file_path):
        if not self.cache.is_cached(file_path):
            data = self._load_file(file_path)
            preprocessed_data = self._preprocess_batch(data)
            self.cache.add_to_cache(file_path, preprocessed_data)
        else:
            preprocessed_data = self.cache.get_from_cache(file_path)
        return preprocessed_data

    def _preprocess_batch(self, batch):
        return self.preprocessor.process_batch(batch)

    def is_dataset_cached(self):
        for file_name in self.file_list:
            file_path = os.path.join(self.data_folder, file_name)
            if not self.cache.is_cached(file_path):
                return False
        return True

    def get_all_data(self):
        if not self.is_dataset_cached():
            for file_name in self.file_list:
                file_path = os.path.join(self.data_folder, file_name)
                if not self.cache.is_cached(file_path):
                    data = self._load_and_preprocess(file_path)
                    self.cache.add_to_cache(file_path, data)

        return [self.cache.get_from_cache(file_path) for file_path in self.file_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, items):
        file_path = self._get_filename(items)
        data = self._load_and_preprocess(file_path)
        return data

    def _get_full_path(self, filename, extension=".npy"):
        return os.path.join(self.data_folder, f"{filename}{extension}")


class ObsDataset(Dataset):
    def __init__(self, config_data, dh, LT, use_cache=True):
        super().__init__(config_data, use_cache)
        self.dh = dh
        self.LT = LT
        self.filename_format = config_data.get('filename_format', "obs{date}_{formatted_index}")

    def _get_filename(self, items):
        formatted_index = (items[1] % self.LT + 1) * self.dh
        return self._get_full_path(self.filename_format.format(date=items[0].replace('-', ''), formatted_index= 0 if formatted_index==24 else formatted_index))

    def _load_file(self, file_path):
        print(file_path)
        return obs_clean(np.load(file_path), self.crop_indices)


class FakeDataset(Dataset):
    def __init__(self, config_data, dh, LT, use_cache=True):
        super().__init__(config_data, use_cache)
        self.dh = dh
        self.LT = LT
        self.filename_format = config_data.get('filename_format', "genFsemble_{date}_{formatted_index}_1000")

    def _get_filename(self, items):
        return self._get_full_path(self.filename_format.format(date=items[0], formatted_index=(items[1] % self.LT + 1) * self.dh))

    def _load_file(self, file_path):
        return np.load(file_path)


class RealDataset(Dataset):
    def __init__(self, config_data, dh, LT, use_cache=True):
        super().__init__(config_data, use_cache)
        self.dh = dh
        self.LT = LT

    def _get_filename(self, items):
        date, index, df0 = items
        names = df0[(df0['Date'] == f"{date}T21:00:00Z") & (df0['LeadTime'] == (index % self.LT + 1) * self.dh - 1)][
            'Name'].to_list()
        file_names = [self._get_full_path(name) for name in names]
        return file_names

    def _load_file(self, file_path):
        arrays = []
        for file_name in file_path:
            data_s = np.expand_dims(np.load(file_name), axis=0)
            arrays.append(data_s)
            data = np.concatenate(arrays, axis=0)
        return data
