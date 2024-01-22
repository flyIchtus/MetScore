from abc import ABC, abstractmethod
import logging
import copy
import metrics.wind_comp as wc
import numpy as np

from configurable import Configurable
#import wind_comp as wc
#import Metscore.useful_funcs as uf


class Metric(ABC):
    isBatched: bool

    def __init__(self, isBatched=False, names=['metric'],
                var_channel=1, obs_var_channel=1,
                var_indices=[0,1,2], real_var_indices=[1,2,3], obs_var_indices=[0,1,2]):
        self.isBatched = isBatched
        self.names = names
        # which channel of the data samples the variable indices are gonna be on. It should be either
        self.var_channel = var_channel 
        self.var_indices = var_indices # which indices to select (for different variables)
        self.real_var_indices = real_var_indices # which indices to select (for different variables, in case of real data)
        self.obs_var_indices = obs_var_indices # which indices to select (for different variables, in case of obs data)
        self.obs_var_channel = obs_var_channel # which channel of the data samples the variable indices are gonna be on. It should be either

    @classmethod
    def fromName(cls, metric):
        logging.debug(f"Creating metric {metric}")
        for subclass in Metric.__subclasses__():
            if subclass.__name__ == metric['type']:
                # metric["is_batched"], **metric['args']
                if 'args' not in metric:
                    metric['args'] = {}
                metric_cls = subclass(name=metric['name'],**metric['args'])
                print(metric_cls._preprocess, metric_cls._calculateCore)
                return metric_cls

        raise Exception(f"Metric {metric['type']} not found, check config file. "
                        f"List of available metrics: {Metric.__subclasses__()}")

    def calculate(self,real_data,fake_data,obs_data, debiasing=None, debiasing_mode=None, conditioning_members=None, threshold=None):
        
        processed_data = self._preprocess(fake_data, real_data, obs_data, debiasing, debiasing_mode, conditioning_members)
        result = self._calculateCore(processed_data, threshold)

        return result

    @abstractmethod
    def _preprocess(self, *args, **kwargs):
        # Common preprocessing logic for all metrics
        pass

    @abstractmethod
    def _calculateCore(self, processed_data, threshold):
        # Specific calculation logic for each metric
        pass

    def isBatched(self):
        return self.isBatched

    def preprocess_cond_obs(self, real_data, fake_data, obs_data, debiasing, debiasing_mode, conditioning_members):
        assert real_data is not None
        assert obs_data is not None
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables

        print(fake_data.shape)
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data # no copy in this case

        if len(self.real_var_indices)!=real_data.shape[self.var_channel]:
            real_data_p = real_data.take(indices=self.real_var_indices, axis=self.var_channel)
        else:
            real_data_p = real_data # no copy in this case
        
        if len(self.obs_var_indices)!=obs_data.shape[self.obs_var_channel]:
            obs_data_p = obs_data.take(indices=self.obs_var_indices, axis=self.obs_var_channel)
        else:
            obs_data_p = obs_data # no copy in this case
        
        
        fake_data_pp = copy.deepcopy(fake_data_p)
        obs_data_pp = copy.deepcopy(obs_data_p[0])
        real_data_pp = copy.deepcopy(real_data_p)

        fake_data_pp[:,0], fake_data_pp[:,1] = wc.computeWindDir(fake_data_pp[:,0], fake_data_pp[:,1])
        real_data_pp[:,0], real_data_pp[:,1] = wc.computeWindDir(real_data_pp[:,0], real_data_pp[:,1])
        
        if debiasing == True:

            fake_data_pp = wc.debiasing(fake_data_pp, real_data_pp, conditioning_members, mode = debiasing_mode)

        # debiaising fake data

        angle_dif = wc.angle_diff(fake_data_pp[:,1], obs_data_pp[1])


        fake_data_pp[:,1] = angle_dif
        obs_data_pp[1,~np.isnan(obs_data_pp[1])] = 0.
        
        return {'real_data': real_data_pp,
                'fake_data': fake_data_pp,
                'obs_data': obs_data_pp}

    def preprocess_dist(self,real_data,fake_data):
        assert real_data is not None
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables        
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data # no copy in this case
        if len(self.real_var_indices)!=real_data.shape[self.var_channel]:
            real_data_p = real_data.take(indices=self.real_var_indices, axis=self.var_channel)
        else:
            real_data_p = real_data # no copy in this case
        return {'real_data': real_data_p,
                'fake_data': fake_data_p}

    def preprocess_standalone(self, fake_data):
        # selecting only the right indices for variables
        # for that we use np.take, which copies data. 
        # While this is surely costly, at first hand we want to do so 
        # because not all metrics might use the same variables        
        if len(self.var_indices)!=fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data # no copy in this case
        
        return fake_data_p