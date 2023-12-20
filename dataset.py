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
        self.preprocessor = Preprocessor.fromConfig(config_data['preprocessor_config'])
        self.cache = MemoryCache(use_cache)
        self.file_list = os.listdir(config_data['data_folder'])
        self.load_data_semaphore = threading.Semaphore()

    #@thread_safe_semaphore
    @abstractmethod
    def load_data(self, file_path):
        # Charger les données depuis le fichier .npy
        # TODO: utilisre une semaphore pour éviter les conflits si jamais (@thread_safe_semaphore)
        #return np.load(file_path)
        pass

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
                    data = self.load_data(file_path)
                    self.cache.add_to_cache(file_path, data)

        return [self.cache.get_from_cache(file_path) for file_path in self.file_list]

    def __iter__(self):
        batch_size = self.batch_size
        for i in range(0, len(self.file_list), batch_size):
            batch_files = self.file_list[i:i + batch_size]
            batch_data = []
            for file in batch_files:
                file_path = os.path.join(self.data_folder, file)
                if not self.cache.is_cached(file_path):
                    data = self._preprocess_batch(self.load_data(file_path))
                    self.cache.add_to_cache(file_path, data)
                else:
                    data = self.cache.get_from_cache(file_path)

                batch_data.append(data)

            yield np.array(batch_data)

class Obs_dataset(Dataset):
        def load_data(self, file_path):
            
            return obs_clean(np.load(file_path), self.crop_indices)#objet porté par obs_dataset

class Fake_dataset(Dataset):
        def load_data(self, file_path):

            return np.load(file_path)

class Real_dataset(Dataset):
        def load_data(self, file_path):

            return np.load(file_path)