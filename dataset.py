import os
import threading

import numpy as np

from configurable import Configurable
from useful_funcs import obs_clean
from abc import ABC, abstractmethod

from datetime import datetime, timedelta

from transforms.preprocessor import Preprocessor
from transforms.rrPreprocessor import ReverserrPreprocessor


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
        
        preprocessor_name = config_data['preprocessor_config']['name']
        preprocessor_class = globals().get(preprocessor_name, None)

        if preprocessor_class is None:
            raise ValueError(f"Invalid preprocessor name: {preprocessor_name}")

        self.preprocessor = preprocessor_class.fromConfig(config_data['preprocessor_config'])
        #print(config_data['preprocessor_config']['name'])
        #self.preprocessor = Preprocessor.fromConfig(config_data['preprocessor_config'])
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
        print(self.preprocessor)
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
    def __init__(self, config_data, dh, LT, start_time, use_cache=True):
        super().__init__(config_data, use_cache)

        self.config_data=config_data
        self.dh = dh
        self.LT = LT
        self.start_time = start_time
        self.filename_format = config_data.get('filename_format', "obs{date}_{formatted_index}")


    def _get_filename(self, items):
        real_hour = self.start_time + (items[1] % self.LT + 1)* self.dh
        date_index = int(np.floor(real_hour/24.))

        date_0 = datetime.strptime(items[0], '%Y-%m-%d')
        next_date_1 = date_0 + timedelta(days=1)
        next_date_2 = date_0 + timedelta(days=2)
        date_1 = next_date_1.strftime('%Y-%m-%d')
        date_2 = next_date_2.strftime('%Y-%m-%d')
        dates=[items[0], date_1, date_2] # considering 45 hours of lead time available, there are three possible observation dates
        # print(dates[date_index].replace('-', ''), real_hour%24)
        return self._get_full_path(self.filename_format.format(date=dates[date_index].replace('-', ''), formatted_index= real_hour%24))

    def _load_file(self, file_path):
        # print(file_path)
        return obs_clean(np.load(file_path), self.crop_indices)

    def get_all_data(self, liste_dates_rep):
        all_data = []
        if not self.is_dataset_cached():
            for idx, date in enumerate(liste_dates_rep[:2]):
                # use __getitem__ to load and preprocess data
                all_data.append(self.__getitem__((date, idx)))
        return np.array(all_data)


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

    def get_all_data(self, liste_dates_rep):
        all_data = []
        if not self.is_dataset_cached():
            for idx, date in enumerate(liste_dates_rep[:2]):
                # use __getitem__ to load and preprocess data
                all_data.append(self.__getitem__((date, idx)))
        
        res = np.array(all_data)
        Shape = res.shape
        res = res.reshape(Shape[0] * Shape[1],Shape[2], Shape[3], Shape[4])
        return res


class RealDataset(Dataset):
    def __init__(self, config_data, dh, LT, df0, use_cache=True):
        super().__init__(config_data, use_cache)
        self.dh = dh
        self.LT = LT
        self.df0 = df0

    def _get_filename(self, items):
        date, index = items
        names = self.df0[(self.df0['Date'] == f"{date}T21:00:00Z") & (self.df0['LeadTime'] == (index % self.LT + 1) * self.dh - 1)][
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

    def get_all_data(self, liste_dates_rep):
        all_data = []
        if not self.is_dataset_cached():
            for idx, date in enumerate(liste_dates_rep[:2]):
                # use __getitem__ to load and preprocess data
                all_data.append(self.__getitem__((date, idx)))
        res = np.array(all_data)
        Shape = res.shape
        res = res.reshape(Shape[0] * Shape[1],Shape[2], Shape[3], Shape[4])
        return res
