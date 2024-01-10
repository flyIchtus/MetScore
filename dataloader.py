import numpy as np
import pandas as pd
from configurable import Configurable
from dataset import ObsDataset, FakeDataset, RealDataset


'''
        df0 = pd.read_csv(self.data_dir_real + 'Large_lt_test_labels.csv')
        
        df1 = pd.read_csv(self.data_dir_real + 'Large_lt_test_labels.csv')
        
        List_dates_unique = df0["Date"].unique().tolist()
        List_dates_inv = df1["Date"].unique().tolist()
        
        List_dates_inv.remove('2021-10-29T21:00:00Z') #31.10.2021 obs missing
        List_dates_inv.remove('2021-10-30T21:00:00Z')
        List_dates_unique.remove('2021-10-29T21:00:00Z') #31.10.2021 obs missing
        List_dates_unique.remove('2021-10-30T21:00:00Z')
        
        List_dates_inv_org = copy.deepcopy(List_dates_inv)
        #List_dates_unique.sort()
        #List_dates_inv_org.sort()
        #List_dates_inv.sort()
        
        for i in range(len(List_dates_unique)):
            List_dates_unique[i]=List_dates_unique[i].replace("T21:00:00Z","")
            List_dates_unique[i]=List_dates_unique[i].replace("-","")
        
        for i in range(len(List_dates_inv)):
            List_dates_inv[i]=List_dates_inv[i].replace("T21:00:00Z","")
            
        ############### Putting all the available observation dates in a list
'''


class DataLoader(Configurable):
    def __init__(self, config_data, use_cache=False):
        """
        Basic dataloader class for loading data from datasets
        config : dict
        """
        super().__init__()
        
        self.df0 = pd.read_csv(config_data['path_to_csv'] + '/'+ config_data['csv_file'])
        df_extract = self.df0[(self.df0['Date']>=config_data['date_start']) & (self.df0['Date']<config_data['date_end'])]

        self.liste_dates = df_extract['Date'].unique().tolist()
        self.liste_dates_repl = [date_string.replace('T21:00:00Z', '') for date_string in self.liste_dates]
        self.liste_dates_rep = [item for item in self.liste_dates_repl for _ in range(config_data['Lead_Times'])]

        self.real_dataset = RealDataset.fromConfig(config_data['real_dataset_config'], use_cache=use_cache)
        self.fake_dataset = FakeDataset.fromConfig(config_data['fake_dataset_config'], use_cache=use_cache)
        self.obs_dataset = ObsDataset.fromConfig(config_data['obs_dataset_config'], use_cache=use_cache)

        self.data_length = min(len(self.real_dataset), len(self.fake_dataset), len(self.obs_dataset))
        self.batch_size = config_data['batch_size']
        self.Lead_Times = config_data['Lead_Times']
        self.dh = config_data['dh']
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
            #fake_samples = [self.fake_dataset[self.current_index + i] for i in range(self.batch_size)]
            fake_samples = [self.fake_dataset.__getitem__(self.liste_dates_rep[self.current_index + i], self.current_index + i, self.Lead_Times, self.dh) for i in range(self.batch_size)]
            obs_samples = [self.obs_dataset.__getitem__(self.liste_dates_rep[self.current_index + i], self.current_index + i, self.Lead_Times, self.dh) for i in range(self.batch_size)]
            real_samples = [self.real_dataset.__getitem__(self.liste_dates_rep[self.current_index + i], self.current_index + i, self.Lead_Times, self.dh, self.df0) for i in range(self.batch_size)]

            print(real_samples[0].shape, obs_samples[0].shape, fake_samples[0].shape)
            print(stop)
            # real_samples, obs_samples = [], []
            # for _, date in fake_samples:
            #     real_samples.append(self.real_dataset[date])
            #     obs_samples.append(self.obs_dataset[date])

            self.current_index += self.batch_size

            return [fake_sample[0] for fake_sample in fake_samples], real_samples, obs_samples
        else:
            raise StopIteration
