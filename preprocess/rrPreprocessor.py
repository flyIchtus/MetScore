import copy
import logging
import os

import numpy as np
import scipy

from preprocess.preprocessor import Preprocessor


class subPreprocessor(Preprocessor):
    def __init__(self, config_data, **kwargs):
        super().__init__(config_data, **kwargs)
        self.config_data = config_data
        self.kwargs = kwargs

    def process_batch(self, batch):
        return batch + self.kwargs['add']


class rrPreprocessor(Preprocessor):
    required_keys = ['real_data_dir', 'real_var_indices', 'normalization']

    def __init__(self, config_data, sizeH, sizeW, variables, **kwargs):
        super().__init__(config_data, **kwargs)
        self.config_data = config_data
        self.sizeH = sizeH
        self.sizeW = sizeW
        self.variables = variables
        self.normalization = config_data['normalization']
        self.crop_size = (self.sizeH, self.sizeW)
        self.maxs, self.mins, self.means, self.stds = self.init_normalization()
        if self.stds is not None:
            self.stds *= 1.0 / 0.95

    def init_normalization(self):
        normalization_type = self.normalization["type"]
        if normalization_type == "mean":
            means, stds = self.load_stat_files(normalization_type, "mean", "std")

            means[1] = 7.13083249e-01
            means[2] = -1.99056945e-01
            means[3] = 2.86256141e+02

            stds[1] = 37.12370493
            stds[2] = 33.48502573
            stds[3] = 46.278571
            logging.warning("Means and stds values are changed manually. we need to fix this")
            return None, None, means, stds
        elif normalization_type == "minmax":
            maxs, mins = self.load_stat_files(normalization_type, "max", "min")
            return maxs, mins, None, None
        elif normalization_type == "quant":
            maxs, mins = self.load_stat_files(normalization_type, "Q99", "Q01")
            return maxs, mins, None, None
        else:
            logging.debug("No normalization set")
            return None, None, None, None

    def load_stat_files(self, normalization_type, str1, str2):
        stat_version = self.config_data["stat_version"]
        log_iterations = self.config_data["rr_transform"]["log_transform_iteration"]
        per_pixel = self.config_data["normalization"]["per_pixel"]

        mean_or_max_filename = f"{str1}_{stat_version}"
        mean_or_max_filename += "_log" * log_iterations
        std_or_min_filename = f"{str2}_{stat_version}"
        std_or_min_filename += "_log" * log_iterations

        if per_pixel:
            mean_or_max_filename += "_ppx"
            std_or_min_filename += "_ppx"
        mean_or_max_filename += ".npy"
        std_or_min_filename += ".npy"
        logging.debug(f"Normalization set to {normalization_type}")
        stat_folder = self.config_data["stat_folder"]
        file_path = os.path.join(self.config_data["real_data_dir"], stat_folder, mean_or_max_filename)
        means_or_maxs = np.load(file_path).astype('float32')
        logging.debug(f"{str1} file found")

        file_path = os.path.join(self.config_data["real_data_dir"], stat_folder, std_or_min_filename)
        stds_or_mins = np.load(file_path).astype('float32')
        logging.debug(f"{str2} file found")
        return means_or_maxs, stds_or_mins

    def detransform(self, data):
        norm_type = self.normalization["type"]
        per_pixel = self.normalization["per_pixel"]
        self.rr_transform = self.rr_transform
        if 'rr' in self.variables and self.self.rr_transform["symetrization"]:
            self.mins = -self.maxs
            self.means = np.zeros_like(self.means)
        if norm_type == "mean":
            if not per_pixel:
                data = data * self.stds[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis] + self.means[
                    np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
            else:
                data = data * self.stds + self.means

        elif norm_type == "minmax" or norm_type == "quant":
            if not per_pixel:
                data = ((data + 1) / 2) * (
                        self.maxs[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis] - self.mins[
                    np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]) + self.mins[
                           np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
            else:
                data = ((data + 1) / 2) * (self.maxs - self.mins) + self.mins

        if 'rr' in self.variables:
            rr_idx = self.real_var_indices[0]
            if self.rr_transform["symetrization"]:
                data[:, rr_idx] = np.abs(data[:, rr_idx])
            for _ in range(self.rr_transform["log_transform_iteration"]):
                try:
                    data[:, rr_idx] = np.exp(data[:, rr_idx]) - 1
                except RuntimeWarning as error:
                    logging.debug(f"RuntimeWarning in np.exp(data[:, rr_idx]) - 1.")
            if self.rr_transform["gaussian_std"] > 0:
                mask_no_rr = data[:, rr_idx] > self.rr_transform["gaussian_std"] * (1 + 0.25)
                data[:, rr_idx] *= mask_no_rr
        logging.debug(f"Data detransformed")
        return data

    def transform(self, data):
        means, stds, maxs, mins = copy.deepcopy(self.means), copy.deepcopy(self.stds), copy.deepcopy(
            self.maxs), copy.deepcopy(self.mins)
        norm_type = self.normalization["type"]
        self.rr_transform = self.self.rr_transform
        if 'rr' in self.variables:
            rr_idx = self.real_var_indices[0]
            for _ in range(self.rr_transform["log_transform_iteration"]):
                data[:, rr_idx] = np.log1p(data[:, rr_idx])
            if self.rr_transform["symetrization"] and np.random.random() <= 0.5:
                data[:, rr_idx] = -data[:, rr_idx]
            if norm_type != "None":
                if self.rr_transform["symetrization"]:  # applying transformations on rr only if selected
                    if norm_type == "means":
                        means[rr_idx] = np.zeros_like(means[rr_idx])
                    elif norm_type == "minmax":
                        mins[rr_idx] = -maxs[rr_idx]
        if norm_type == "mean":
            if np.ndim(stds) > 1 and 'rr' in self.variables:
                if self.normalization["for_rr"]["blur_iteration"] > 0:
                    gaussian_filter = np.float32([[1, 4, 6, 4, 1],
                                                  [4, 16, 24, 16, 4],
                                                  [6, 24, 36, 24, 6],
                                                  [4, 16, 24, 16, 4],
                                                  [1, 4, 6, 4, 1]]) / 256.0
                    for _ in range(self.normalization["for_rr"]["blur_iteration"]):
                        stds[rr_idx] = scipy.ndimage.convolve(stds[rr_idx], gaussian_filter, mode='mirror')
            else:
                means = means[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
                stds = stds[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
        elif norm_type == "minmax":
            logging.debug(maxs.shape)
            if np.ndim(maxs) > 1 and 'rr' in self.variables:
                if self.normalization["for_rr"]["blur_iteration"] > 0:
                    gaussian_filter = np.float32([[1, 4, 6, 4, 1],
                                                  [4, 16, 24, 16, 4],
                                                  [6, 24, 36, 24, 6],
                                                  [4, 16, 24, 16, 4],
                                                  [1, 4, 6, 4, 1]]) / 256.0
                    for _ in range(self.normalization["for_rr"]["blur_iteration"]):
                        maxs[rr_idx] = scipy.ndimage.convolve(maxs[rr_idx], gaussian_filter, mode='mirror')
            else:
                maxs = maxs[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
                mins = mins[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
        if norm_type == "means":
            data = (data - means) / stds
        elif norm_type == "minmax":
            data = -1 + 2 * ((data - mins) / (maxs - mins))
        logging.debug("Real samples transformed...")
        return data


class ForwardrrPreprocessor(rrPreprocessor):
    def __init__(self, config_data, sizeH, sizeW, variables, **kwargs):
        super().__init__(config_data, sizeH, sizeW, variables, **kwargs)

    def process_batch(self, batch):
        self.transform(batch)


class ReverserrPreprocessor(rrPreprocessor):
    def __init__(self, config_data, sizeH, sizeW, variables, **kwargs):
        super().__init__(config_data, sizeH, sizeW, variables, **kwargs)

    def process_batch(self, batch):
        return self.detransform(batch)
