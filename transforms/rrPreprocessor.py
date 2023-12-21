import numpy as np
import os
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import preprocessor as Pre


class rrPreprocessor(Pre.Preprocessor):
    def __init__(self, config, crop_size):
        super.__init__()
        self.config = config
        self.crop_size = crop_size
        self.dataset_handler_yaml = config.data_transform_config
        self.maxs, self.mins, self.means, self.stds = self.init_normalization()
        if self.stds is not None:
            self.stds *= 1.0 / 0.95

    def init_normalization(self):
        normalization_type = self.dataset_handler_yaml["normalization"]["type"]
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
        stat_version = self.dataset_handler_yaml["stat_version"]
        log_iterations = self.dataset_handler_yaml["rr_transform"]["log_transform_iteration"]
        per_pixel = self.dataset_handler_yaml["normalization"]["per_pixel"]

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
        stat_folder = self.dataset_handler_yaml["stat_folder"]
        file_path = os.path.join(real_data_dir, stat_folder, mean_or_max_filename)
        means_or_maxs = np.load(file_path).astype('float32')
        print(f"{str1} file found")

        file_path = os.path.join(real_data_dir, stat_folder, std_or_min_filename)
        stds_or_mins = np.load(file_path).astype('float32')
        print(f"{str2} file found")
        return means_or_maxs, stds_or_mins

    def detransform(self, data, step, VI):
        norm_type = self.dataset_handler_yaml["normalization"]["type"]
        per_pixel = self.dataset_handler_yaml["normalization"]["per_pixel"]
        rr_transform = self.dataset_handler_yaml["rr_transform"]
        if rr_transform["symetrization"]:
            self.mins = -self.maxs
            self.means = np.zeros_like(self.means)
        if norm_type == "mean":
            if not per_pixel:
                data = data * self.stds[np.newaxis, VI, np.newaxis, np.newaxis] + self.means[np.newaxis, VI, np.newaxis, np.newaxis]
            else:
                data = data * self.stds + self.means
        elif norm_type == "minmax" or norm_type == "quant":
            if not per_pixel:
                data = ((data + 1) / 2) * (self.maxs[np.newaxis, VI, np.newaxis, np.newaxis] - self.mins[np.newaxis, VI, np.newaxis, np.newaxis]) + self.mins[np.newaxis, VI, np.newaxis, np.newaxis]
            else:
                data = ((data + 1) / 2) * (self.maxs - self.mins) + self.mins
        if rr_transform["symetrization"]:
            data[:, 0] = np.abs(data[:, 0])
        for _ in range(rr_transform["log_transform_iteration"]):
            try:
                data[:, 0] = np.exp(data[:, 0]) - 1
            except RuntimeWarning as error:
                print(f"RuntimeWarning for step {step}, in np.exp(data[:, 0]) - 1.")
        if rr_transform["gaussian_std"] > 0:
            mask_no_rr = data[:, 0] > rr_transform["gaussian_std"] * (1 + 0.25)
            data[:, 0] *= mask_no_rr
        print("Detransform OK.")
        return data

    def print_data_detransf(self, data, step, VI, VI_f):
        numpy_files = [os.path.join(real_data_dir, x) for x in os.listdir(real_data_dir) if x.endswith(".npy")]
        random_files = np.random.choice(numpy_files, size=4, replace=False)
        reals = np.array([np.load(file) for file in random_files])
        data_to_print = np.concatenate((data[:12, VI_f], reals[:,VI]), axis=0)
        save_dir = f"{self.config.data_dir_f[:-1]}_detranformed/"
        os.makedirs(save_dir, exist_ok=True)

        print("Saving fake samples...")
        np.save(f"{save_dir}Samples_at_Step_{step}.npy", data[:12])

        print("Printing data...")
        self.online_sample_plot(data_to_print, step)

        print("Data printed.")
    
    def transform(self, data, VI):
        means, stds, maxs, mins = copy.deepcopy(self.means), copy.deepcopy(self.stds), copy.deepcopy(self.maxs), copy.deepcopy(self.mins)
        norm_type = self.dataset_handler_yaml["normalization"]["type"]
        rr_transform = self.dataset_handler_yaml["rr_transform"]
        for _ in range(rr_transform["log_transform_iteration"]):
            data[:, 0] = np.log1p(data[:, 0])
        if rr_transform["symetrization"] and np.random.random() <= 0.5:
            data[:, 0] = -data[:, 0]
        if norm_type != "None":
            if rr_transform["symetrization"]: #applying transformations on rr only if selected
                if norm_type == "means":
                    means[0] = np.zeros_like(means[0])
                elif norm_type == "minmax":
                    mins[0] = -maxs[0]
        # gaussian_std = rr_transform["gaussian_std"]
        # if gaussian_std:
        #     for _ in range(rr_transform["log_transform_iteration"]):
        #         gaussian_std = np.log(1 + gaussian_std)
        #     gaussian_std_map = np.random.choice([-1, 1], size=self.crop_size) * gaussian_std
        #     gaussian_noise = np.mod(np.random.normal(0, gaussian_std, size=self.crop_size), gaussian_std_map)
        if norm_type == "mean":
            if np.ndim(stds) > 1:
                if self.dataset_handler_yaml["normalization"]["for_rr"]["blur_iteration"] > 0:
                    gaussian_filter = np.float32([[1, 4,  6,  4,  1],
                                                [4, 16, 24, 16, 4],
                                                [6, 24, 36, 24, 6],
                                                [4, 16, 24, 16, 4],
                                                [1, 4,  6,  4,  1]]) / 256.0
                    for _ in range(self.dataset_handler_yaml["normalization"]["for_rr"]["blur_iteration"]):
                        stds[0] = scipy.ndimage.convolve(stds[0], gaussian_filter, mode='mirror')
            else:
                means = means[np.newaxis, :, np.newaxis, np.newaxis]
                stds = stds[np.newaxis, :, np.newaxis, np.newaxis]
        elif norm_type == "minmax":
            print(maxs.shape)
            if np.ndim(maxs) > 1:
                if self.dataset_handler_yaml["normalization"]["for_rr"]["blur_iteration"] > 0:
                    gaussian_filter = np.float32([[1, 4,  6,  4,  1],
                                                [4, 16, 24, 16, 4],
                                                [6, 24, 36, 24, 6],
                                                [4, 16, 24, 16, 4],
                                                [1, 4,  6,  4,  1]]) / 256.0
                    for _ in range(self.dataset_handler_yaml["normalization"]["for_rr"]["blur_iteration"]):
                        maxs[0] = scipy.ndimage.convolve(maxs[0], gaussian_filter, mode='mirror')
            else:
                maxs = maxs[np.newaxis, VI, np.newaxis, np.newaxis]
                mins = mins[np.newaxis, VI, np.newaxis, np.newaxis]
        # if gaussian_std != 0:
        #     mask_no_rr = (data[:, 0] <= gaussian_std)
        #     data[:, 0] = data[:, 0] + gaussian_noise * mask_no_rr
        if norm_type == "means":
            data = (data - means) / stds
        elif norm_type == "minmax":
            data = -1 + 2 * ((data - mins) / (maxs - mins))
        print("Real samples transformed...")
        return data
        

    def online_sample_plot(self, batch, Step, mean_pert=False):
        bounds = np.array([0, 0.5, 1, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 250, 300, 350, 1000])
        # bounds = np.array([0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100])
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=18)
        cmapRR = colors.ListedColormap(["white", "#63006e", "#0000ff", "#00b2ff", "#00ffff", "#08dfd6", "#1cb8a5", "#6ba530", "#ffff00", "#ffd800", "#ffa500", "#ff0000", "#991407", "#ff00ff", "#a4ff00", "#00fa00", "#31be00", "#31858b"], name="from_list", N=None)
        batch_to_print = batch[:16]
        IMG_SIZE = batch.shape[2]
        
        variable_mapping = {
            "rr": ("Rain rate", cmapRR, (0, 1000)),
            "u": ("Wind u", "viridis", (-20, 20)),
            "v": ("Wind v", "viridis", (-20, 20)),
            "t2m": ("2m temperature", "coolwarm", (240, 316)),
            "orog": ("Orography", "terrain", (-0.95, 0.95)),
            "z500": ("500 hPa geopotential", "Blues", (0, 100)),
            "t850": ("850 hPa temperature", "coolwarm", (-0.5, 0.5)),
            "tpw850": ("tpw850", "plasma", (-0.5, 0.5)),
        }

        for i, var in enumerate(self.config.variables):
            varname, cmap, limits = variable_mapping.get(var)

            fig, axs = plt.subplots(4, 4, figsize=(20, 20))
            st = fig.suptitle(f"{varname}{' pert' if mean_pert else ''}", fontsize='30')
            st.set_y(0.96)

            for j, ax in enumerate(axs.ravel()):
                b = batch_to_print[j][i]
                if var == "rr":
                    im = ax.imshow(b[::-1, :], cmap=cmap, norm=norm)
                else:
                    im = ax.imshow(b[::-1, :], cmap=cmap, vmin=limits[0], vmax=limits[1])

            fig.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.9)
            cbax = fig.add_axes([0.92, 0.05, 0.02, 0.85])
            cb = fig.colorbar(im, cax=cbax)
            cb.ax.tick_params(labelsize=20)

            save_dir = f"{self.config.data_dir_f[:-1]}_detranformed/"
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}Samples_at_Step_{Step}_{var}{'_pert' if mean_pert else ''}.png")
            plt.close()

            if mean_pert:
                fig, axs = plt.subplots(4, 4, figsize=(20, 20))
                st = fig.suptitle(f"{varname} mean", fontsize='30')
                st.set_y(0.96)

                for j, ax in enumerate(axs.ravel()):
                    b = batch_to_print[j][i + len(self.config.variables)].view(IMG_SIZE, IMG_SIZE)
                    if var == "rr":
                        im = ax.imshow(b[::-1, :], cmap=cmap, norm=norm)
                    else:
                        im = ax.imshow(b[::-1, :], cmap=cmap, vmin=limits[0], vmax=limits[1])

                fig.subplots_adjust(bottom=0.05, top=0.9, left=0.05, right=0.9)
                cbax = fig.add_axes([0.92, 0.05, 0.02, 0.85])
                cb = fig.colorbar(im, cax=cbax)
                cb.ax.tick_params(labelsize=20)

                plt.savefig(f"{save_dir}Samples_at_Step_{Step}_{var}{'_mean' if mean_pert else ''}.png")
                plt.close()