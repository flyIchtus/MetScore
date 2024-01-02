import numpy as np
import os
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import preprocessor as Pre
import yaml


class rrPreprocessor(Pre.Preprocessor):
    def __init__(self, config_data):
        super.__init__()
        self.config_dir = config_data['configDir']
        with open(self.config_dir + config_data['configFile'], 'r') as f:
            self.config_yaml = yaml.safe_load(f)
        self.crop_size = (self.config_yaml.sizeH, self.config_yaml.sizeW)
        self.real_var_indices = config_data['real_var_indices']
        self.maxs, self.mins, self.means, self.stds = self.init_normalization()
        if self.stds is not None:
            self.stds *= 1.0 / 0.95

    def init_normalization(self):
        normalization_type = self.config_yaml["normalization"]["type"]
        if normalization_type == "mean":
            means, stds = self.load_stat_files(normalization_type, "mean", "std")
            return None, None, means, stds
        elif normalization_type == "minmax":
            maxs, mins = self.load_stat_files(normalization_type, "max", "min")
            return maxs, mins, None, None
        elif normalization_type == "quant":
            maxs, mins = self.load_stat_files(normalization_type, "Q99", "Q01")
            return maxs, mins, None, None
        else:
            print("No normalization set")
            return None, None, None, None

    def load_stat_files(self, normalization_type, str1, str2):
        stat_version = self.config_yaml["stat_version"]
        log_iterations = self.config_yaml["rr_transform"]["log_transform_iteration"]
        per_pixel = self.config_yaml["normalization"]["per_pixel"]

        mean_or_max_filename = f"{str1}_{stat_version}"
        mean_or_max_filename += "_log" * log_iterations
        std_or_min_filename = f"{str2}_{stat_version}"
        std_or_min_filename += "_log" * log_iterations

        if per_pixel:
            mean_or_max_filename += "_ppx"
            std_or_min_filename += "_ppx"
        mean_or_max_filename += ".npy"
        std_or_min_filename += ".npy"
        print(f"Normalization set to {normalization_type}")
        stat_folder = self.config_yaml["stat_folder"]
        file_path = os.path.join(real_data_dir, stat_folder, mean_or_max_filename)
        means_or_maxs = np.load(file_path).astype('float32')
        print(f"{str1} file found")

        file_path = os.path.join(real_data_dir, stat_folder, std_or_min_filename)
        stds_or_mins = np.load(file_path).astype('float32')
        print(f"{str2} file found")
        return means_or_maxs, stds_or_mins

    def detransform(self, data):
        norm_type = self.config_yaml["normalization"]["type"]
        per_pixel = self.config_yaml["normalization"]["per_pixel"]
        rr_transform = self.config_yaml["rr_transform"] 
        if 'rr' in self.config_data['variables'] and rr_transform["symetrization"]:
            self.mins = -self.maxs
            self.means = np.zeros_like(self.means)
        if norm_type == "mean":
            if not per_pixel:
                data = data * self.stds[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis] + self.means[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
            else:
                data = data * self.stds + self.means
        elif norm_type == "minmax" or norm_type == "quant":
            if not per_pixel:
                data = ((data + 1) / 2) * (self.maxs[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis] - self.mins[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]) + self.mins[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
            else:
                data = ((data + 1) / 2) * (self.maxs - self.mins) + self.mins
        if 'rr' in self.config_data['variables']:
            rr_idx = self.real_var_indices[0]
            if rr_transform["symetrization"]:
                data[:, rr_idx] = np.abs(data[:, rr_idx])
            for _ in range(rr_transform["log_transform_iteration"]):
                try:
                    data[:, rr_idx] = np.exp(data[:, rr_idx]) - 1
                except RuntimeWarning as error:
                    print(f"RuntimeWarning in np.exp(data[:, rr_idx]) - 1.")
            if rr_transform["gaussian_std"] > 0:
                mask_no_rr = data[:, rr_idx] > rr_transform["gaussian_std"] * (1 + 0.25)
                data[:, rr_idx] *= mask_no_rr
        print("Detransform OK.")
        return data
    
    def transform(self, data):
        means, stds, maxs, mins = copy.deepcopy(self.means), copy.deepcopy(self.stds), copy.deepcopy(self.maxs), copy.deepcopy(self.mins)
        norm_type = self.config_yaml["normalization"]["type"]
        rr_transform = self.config_yaml["rr_transform"]
        if 'rr' in self.config_data['variables']:
            rr_idx = self.real_var_indices[0]
            for _ in range(rr_transform["log_transform_iteration"]):
                data[:, rr_idx] = np.log1p(data[:, rr_idx])
            if rr_transform["symetrization"] and np.random.random() <= 0.5:
                data[:, rr_idx] = -data[:, rr_idx]
            if norm_type != "None":
                if rr_transform["symetrization"]: #applying transformations on rr only if selected
                    if norm_type == "means":
                        means[rr_idx] = np.zeros_like(means[rr_idx])
                    elif norm_type == "minmax":
                        mins[rr_idx] = -maxs[rr_idx]
        # gaussian_std = rr_transform["gaussian_std"]
        # if gaussian_std:
        #     for _ in range(rr_transform["log_transform_iteration"]):
        #         gaussian_std = np.log(1 + gaussian_std)
        #     gaussian_std_map = np.random.choice([-1, 1], size=self.crop_size) * gaussian_std
        #     gaussian_noise = np.mod(np.random.normal(0, gaussian_std, size=self.crop_size), gaussian_std_map)
        if norm_type == "mean" :
            if np.ndim(stds) > 1 and 'rr' in self.config_data['variables']:
                if self.config_yaml["normalization"]["for_rr"]["blur_iteration"] > 0:
                    gaussian_filter = np.float32([[1, 4,  6,  4,  1],
                                                [4, 16, 24, 16, 4],
                                                [6, 24, 36, 24, 6],
                                                [4, 16, 24, 16, 4],
                                                [1, 4,  6,  4,  1]]) / 256.0
                    for _ in range(self.config_yaml["normalization"]["for_rr"]["blur_iteration"]):
                        stds[rr_idx] = scipy.ndimage.convolve(stds[rr_idx], gaussian_filter, mode='mirror')
            else:
                means = means[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
                stds = stds[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
        elif norm_type == "minmax":
            print(maxs.shape)
            if np.ndim(maxs) > 1 and 'rr' in self.config_data['variables']:
                if self.config_yaml["normalization"]["for_rr"]["blur_iteration"] > 0:
                    gaussian_filter = np.float32([[1, 4,  6,  4,  1],
                                                [4, 16, 24, 16, 4],
                                                [6, 24, 36, 24, 6],
                                                [4, 16, 24, 16, 4],
                                                [1, 4,  6,  4,  1]]) / 256.0
                    for _ in range(self.config_yaml["normalization"]["for_rr"]["blur_iteration"]):
                        maxs[rr_idx] = scipy.ndimage.convolve(maxs[rr_idx], gaussian_filter, mode='mirror')
            else:
                maxs = maxs[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
                mins = mins[np.newaxis, self.real_var_indices, np.newaxis, np.newaxis]
        # if gaussian_std != 0:
        #     mask_no_rr = (data[:, 0] <= gaussian_std)
        #     data[:, 0] = data[:, 0] + gaussian_noise * mask_no_rr
        if norm_type == "means":
            data = (data - means) / stds
        elif norm_type == "minmax":
            data = -1 + 2 * ((data - mins) / (maxs - mins))
        print("Real samples transformed...")
        return data
    
class ForwardrrPreprocessor(rrPreprocessor):
    def __init__(self,config_data):
        super().__init__(config_data)
    
    def process_batch(batch):
        self.transform(batch)

class ReverserrPreprocessor(rrPreprocessor):
    def __init__(self,config_data):
        super().__init__(config_data)
    
    def process_batch(batch):
        self.detransform(batch)