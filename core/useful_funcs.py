import numpy as np

import numpy as np

def obs_clean(obs, crop_indices):
    """
    Clean and process observation data from a given file.

    This function sorts the observations by longitude, filters the observations based on the specified latitude and
    longitude boundaries, averages duplicate observations, and creates an observation matrix with the same shape as
    the fake/real ensemble.

    Parameters:
        obs (np.ndarray): Observation data of shape (N_obs, 3), where the first column represents longitude, the second
                          column represents latitude, and the third column represents the measurement.
        crop_indices (np.ndarray): An array of shape (4,) specifying the indices of the top-left and bottom-right
                                   corners of the cropped area in the AROME grid.

    Returns:
        np.ndarray: A cleaned and processed observation matrix of shape (3, size, size), where the first channel
                    corresponds to the third column of the input observations, the second channel corresponds to the
                    second column, and the third channel corresponds to the first column.
    """
    ind = np.argsort(obs[:, 0])
    obs = obs[ind]

    N_obs = obs.shape[0]

    Lat_min_AROME = 37.5
    Lat_max_AROME = 55.4
    Lon_min_AROME = -12.0
    Lon_max_AROME = 16.0
    n_lat_AROME = 717
    n_lon_AROME = 1121

    size = crop_indices[1] - crop_indices[0]

    Lat_min = Lat_min_AROME + crop_indices[0] * (Lat_max_AROME - Lat_min_AROME) / n_lat_AROME
    Lat_max = Lat_min_AROME + crop_indices[1] * (Lat_max_AROME - Lat_min_AROME) / n_lat_AROME
    Lon_min = Lon_min_AROME + crop_indices[2] * (Lon_max_AROME - Lon_min_AROME) / n_lon_AROME
    Lon_max = Lon_min_AROME + crop_indices[3] * (Lon_max_AROME - Lon_min_AROME) / n_lon_AROME

    dlat = (Lat_max - Lat_min) / size
    dlon = (Lon_max - Lon_min) / size

    obs_reduced = []
    indices_obs = []

    for i in range(N_obs):
        if (obs[i, 0] > Lon_min and obs[i, 0] < Lon_max) and (obs[i, 1] > Lat_min and obs[i, 1] < Lat_max):
            indice_lon = np.floor((obs[i, 0] - Lon_min) / dlon)
            indice_lat = np.floor((obs[i, 1] - Lat_min) / dlat)
            indices_obs.append([indice_lat, indice_lon])
            obs_reduced.append(obs[i])

    indices_obs = np.array(indices_obs, dtype='int')
    obs_reduced = np.array(obs_reduced, dtype='float32')

    len_obs_reduced = obs_reduced.shape[0]

    obs_r_clean = []
    indices_o_clean = []
    j = 0
    sum_measurements = np.zeros((3))
    for i in range(len_obs_reduced):
        if (i == j):
            sum_measurements = sum_measurements + obs_reduced[i, 2::]
            j = i + 1
            if i != len_obs_reduced - 1:
                while (j < len_obs_reduced) and (
                        indices_obs[i, 0] == indices_obs[j, 0] and indices_obs[i, 1] == indices_obs[j, 1]):
                    sum_measurements = sum_measurements + obs_reduced[j, 2::]
                    j = j + 1

                observation = sum_measurements / (j - i)
                sum_measurements = np.zeros((3))
                obs_r_clean.append(observation)
                indices_o_clean.append(indices_obs[i])

    indices_o_clean = np.array(indices_o_clean, dtype='int')
    obs_r_clean = np.array(obs_r_clean, dtype='float32')

    Ens_observation = np.empty((3, size, size))
    Ens_observation[:] = np.nan

    Ens_observation[0, indices_o_clean[:, 0], indices_o_clean[:, 1]] = obs_r_clean[:, 2]
    Ens_observation[1, indices_o_clean[:, 0], indices_o_clean[:, 1]] = obs_r_clean[:, 1]
    Ens_observation[2, indices_o_clean[:, 0], indices_o_clean[:, 1]] = obs_r_clean[:, 0]

    Ens_observation[Ens_observation > 1000.] = np.nan
    Ens_observation[1][Ens_observation[0] < 2.] = np.nan

    return Ens_observation


def denorm(mat, Maxs, Means):
    res = mat * (1. / 0.95) * Maxs.astype('float32') + Means.astype('float32')
    return res
