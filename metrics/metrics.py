# Importations
from abc import ABC, abstractmethod
import logging
import copy
import metrics.wind_comp as wc
import numpy as np

from configurable import Configurable


# Classe Metric
class Metric(ABC, Configurable):

    required_keys = ['name']

    def __init__(self, isBatched=False, **kwargs):
        # self.isBatched = kwargs.get('isBatched', False)
        self.isBatched = isBatched
        super().__init__()


    def calculate(self, real_data, fake_data, obs_data):
        processed_data = self._preprocess(real_data, fake_data, obs_data)
        result = self._calculateCore(processed_data)
        return result

    @abstractmethod
    def _preprocess(self, real_data=None, fake_data=None, obs_data= None):
        pass

    @abstractmethod
    def _calculateCore(self, processed_data):
        pass

    def isBatched(self):
        return self.isBatched


class PreprocessCondObs(Metric):
    def _preprocess(self, real_data=None, fake_data=None, obs_data= None):
        assert real_data is not None
        assert obs_data is not None

        logging.debug(fake_data.shape)
        if len(self.var_indices) != fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data

        if len(self.real_var_indices) != real_data.shape[self.var_channel]:
            real_data_p = real_data.take(indices=self.real_var_indices, axis=self.var_channel)
        else:
            real_data_p = real_data

        if len(self.obs_var_indices) != obs_data.shape[self.obs_var_channel]:
            obs_data_p = obs_data.take(indices=self.obs_var_indices, axis=self.obs_var_channel)
        else:
            obs_data_p = obs_data

        fake_data_pp = copy.deepcopy(fake_data_p)
        obs_data_pp = copy.deepcopy(obs_data_p[0])
        real_data_pp = copy.deepcopy(real_data_p)

        fake_data_pp[:, 0], fake_data_pp[:, 1] = wc.computeWindDir(fake_data_pp[:, 0], fake_data_pp[:, 1])
        real_data_pp[:, 0], real_data_pp[:, 1] = wc.computeWindDir(real_data_pp[:, 0], real_data_pp[:, 1])

        if self.debiasing == True:
            fake_data_pp = wc.debiasing(fake_data_pp, real_data_pp, self.conditioning_members, mode=self.debiasing_mode)

        angle_dif = wc.angle_diff(fake_data_pp[:, 1], obs_data_pp[1])
        fake_data_pp[:, 1] = angle_dif
        obs_data_pp[1, ~np.isnan(obs_data_pp[1])] = 0.

        return {'real_data': real_data_pp,
                'fake_data': fake_data_pp,
                'obs_data': obs_data_pp}

class PreprocessDist(Metric):
    def _preprocess(self, real_data=None, fake_data=None, obs_data= None):
        assert real_data is not None
        if len(self.var_indices) != fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data
        if len(self.real_var_indices) != real_data.shape[self.var_channel]:
            real_data_p = real_data.take(indices=self.real_var_indices, axis=self.var_channel)
        else:
            real_data_p = real_data
        return {'real_data': real_data_p,
                'fake_data': fake_data_p}


class PreprocessStandalone(Metric):
    def _preprocess(self, real_data=None, fake_data=None, obs_data= None):
        if len(self.var_indices) != fake_data.shape[self.var_channel]:
            fake_data_p = fake_data.take(indices=self.var_indices, axis=self.var_channel)
        else:
            fake_data_p = fake_data
        return fake_data_p
