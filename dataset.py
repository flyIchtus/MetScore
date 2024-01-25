import logging
import os
import re
import threading

import numpy as np
import pandas as pd

from configurable import Configurable
from useful_funcs import obs_clean
from abc import ABC, abstractmethod

from datetime import datetime, timedelta

from transforms.preprocessor import Preprocessor

def convert_key(func):
    def wrapper(self, key, *args, **kwargs):
        if type(key) == list:
            fusion_key = key[0]
            for k in key[1:]:
                assert type(k) == str
                fusion_key += k
            key = fusion_key
        return func(self, key, *args, **kwargs)

    return wrapper

class MemoryCache:
    def __init__(self, use_cache):
        self.cache = {}
        self.use_cache = use_cache

    @convert_key
    def add_to_cache(self, key, data):
        if self.use_cache:
            self.cache[key] = data

    @convert_key
    def is_cached(self, key):
        if not self.use_cache:
            return False
        return key in self.cache

    @convert_key
    def get_from_cache(self, key):
        if not self.use_cache:
            return None
        return self.cache[key]

    def clear_cache(self):
        self.cache = {}


class Dataset(Configurable):

    required_keys = ['data_folder', 'preprocessor_config']

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
        self.preprocessor = Preprocessor.from_typed_config(config_data['preprocessor_config'], **config_data)
        logging.debug(f"Using preprocessor: {self.preprocessor.type}")
        self.cache = MemoryCache(use_cache)
        self.load_data_semaphore = threading.Semaphore()

    @abstractmethod
    def _get_filename(self, index):
        pass

    @abstractmethod
    def _load_file(self, file_path):
        pass

    @abstractmethod
    def __len__(self):
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
        for idx in range(len(self)):
            file_path = self._get_filename(idx)
            if not self.cache.is_cached(file_path):
                return False
        return True

    def get_all_data(self):
        all_data = []
        if not self.is_dataset_cached():
            for idx in range(len(self)):
                file_path = self._get_filename(idx)
                data = self._load_and_preprocess(file_path)
                all_data.append(data)
        else:
            for idx in range(len(self)):
                file_path = self._get_filename(idx)
                data = self.cache.get_from_cache(file_path)
                all_data.append(data)
        return np.array(all_data)

    def __getitem__(self, items):
        file_path = self._get_filename(items)
        data = self._load_and_preprocess(file_path)
        return data

    def _get_full_path(self, filename, extension=".npy"):
        return os.path.join(self.data_folder, f"{filename}{extension}")

class DateDataset(Dataset):

    required_keys = ['data_folder', 'preprocessor_config', 'crop_indices']

    def __init__(self, config_data, use_cache=True):
        super().__init__(config_data, use_cache)
        self.df0 = pd.read_csv(os.path.join(config_data['path_to_csv'], config_data['csv_file']))
        df_extract = self.df0[
            (self.df0['Date'] >= config_data['date_start']) & (self.df0['Date'] < config_data['date_end'])]
        self.liste_dates = df_extract['Date'].unique().tolist()
        self.liste_dates = self.liste_dates[0:config_data['number_of_dates']]
        self.liste_dates_repl = [date_string.replace('T21:00:00Z', '') for date_string in self.liste_dates]
        self.liste_dates_rep = [item for item in self.liste_dates_repl for _ in range(config_data['Lead_Times'])]

    def _get_filename(self, index):
        raise NotImplementedError("Subclasses must implement this method.")

    def _load_file(self, file_path):
        raise NotImplementedError("Subclasses must implement this method.")

    def __len__(self):
        return len(self.liste_dates_rep)


class ObsDataset(DateDataset):
    def __init__(self, config_data, use_cache=True):
        super().__init__(config_data, use_cache)
        self.filename_format = config_data.get('filename_format', "obs{date}_{formatted_index}")

    def _get_filename(self, index):
        format_variables = [var.strip('}{') for var in re.findall(r'{(.*?)}', self.filename_format)]
        kwargs = {}

        real_hour = self.start_time + (index % self.Lead_Times + 1) * self.dh

        if 'formatted_index' in format_variables:
            format_variables.remove('formatted_index')
            formatted_index = real_hour % 24
            kwargs = {'formatted_index': formatted_index}

        if 'date' in format_variables:
            format_variables.remove('date')
            date = self.liste_dates_rep[index]
            date_index = int(np.floor(real_hour / 24.))
            date_0 = datetime.strptime(date, '%Y-%m-%d')
            next_date_1 = date_0 + timedelta(days=1)
            next_date_2 = date_0 + timedelta(days=2)
            date_1 = next_date_1.strftime('%Y-%m-%d')
            date_2 = next_date_2.strftime('%Y-%m-%d')
            dates = [date, date_1, date_2]
            kwargs = kwargs | {'date': dates[date_index].replace('-', '')}

        kwargs = kwargs | {var: getattr(self, var, '') for var in format_variables}

        return self._get_full_path(
            self.filename_format.format(**kwargs)
        )

    def _load_file(self, file_path):
        return obs_clean(np.load(file_path), self.crop_indices)


class FakeDataset(DateDataset):
    def __init__(self, config_data, use_cache=True):
        super().__init__(config_data, use_cache)

        self.filename_format = config_data.get('filename_format', "genFsemble_{date}_{formatted_index}_{inv_step}_{cond_members}_{N_ens}")

    def _get_filename(self, index):
        format_variables = [var.strip('}{') for var in re.findall(r'{(.*?)}', self.filename_format)]
        kwargs = {}

        if 'formatted_index' in format_variables:
            format_variables.remove('formatted_index')
            formatted_index = (index % self.Lead_Times + 1) * self.dh
            kwargs = {'formatted_index': formatted_index}

        if 'date' in format_variables:
            format_variables.remove('date')
            date = self.liste_dates_rep[index]
            kwargs = kwargs | {'date': date}

        kwargs = kwargs | {var: getattr(self, var, '') for var in format_variables}

        return self._get_full_path(
            self.filename_format.format(**kwargs)
        )

    def _load_file(self, file_path):
        return np.load(file_path)


class RealDataset(DateDataset):
    def _get_filename(self, index):
        date = self.liste_dates_rep[index]
        names = self.df0[
            (self.df0['Date'] == f"{date}T21:00:00Z") & (
                        self.df0['LeadTime'] == (index % self.Lead_Times + 1) * self.dh - 1)][
            'Name'].to_list()
        file_names = [self._get_full_path(name) for name in names]
        return file_names

    def _load_file(self, file_path):
        arrays = [np.expand_dims(np.load(file_name), axis=0) for file_name in file_path]
        return np.concatenate(arrays, axis=0)
