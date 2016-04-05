import pickle
import numpy as np
import scipy.io as sio
from scipy import sparse as sp

num_subjects = 9
num_words_per_cond = 5
num_conds = 12
dim_x = 51
dim_y = 61
dim_z = 23

file_X = "x_sparse.p"
file_Y = "y.p"


def load_data(from_stored_data=False):
    """ Loads all subjects' .mat files and concatenates them into one whole data set and label set.

    :param from_stored_data: True if no need to create the sets. Will load pre-created ones within same directory
    :return: data set data_X and label set data_Y
    """

    if from_stored_data:
        data_X = pickle.load(open(file_X, "rb"))
        data_Y = pickle.load(open(file_Y, "rb"))
        return data_X, data_Y

    data_X = None
    data_Y = None

    for num_subject in range(num_subjects):
        subject_data = sio.loadmat("data/data-science-P" + str(num_subject + 1) + ".mat")

        # big three headers
        meta = subject_data.get("meta")
        info = subject_data.get("info")[0]
        trials = subject_data.get("data")

        # meta data
        nvoxels = meta["nvoxels"][0][0][0][0]
        colToCoord = meta["colToCoord"][0][0]
        coordToCol = meta["coordToCol"][0][0]

        for num_trial in range(len(trials)):
            # create feature vectors
            voxels = trials[num_trial][0][0]
            feature_vec = np.zeros(dim_x * dim_y * dim_z)
            for i in range(len(voxels)):
                colInfo = colToCoord[i, :]
                x = colInfo[0] - 1  # index in data starts from 1
                y = colInfo[1] - 1  # same
                z = colInfo[2] - 1  # same
                feature_vec[z * (dim_x * dim_y) + y * dim_x + x] = voxels[i]
            feature_vec = sp.csr_matrix(feature_vec)

            # create label vectors
            trial_info = info[num_trial]
            cond_number = trial_info[1][0][0] - 2  # starts from 2 (2 ~ 13)
            word_number = trial_info[3][0][0] - 1  # starts from 1 (1 ~ 5)
            label_vec = np.zeros(num_conds * num_words_per_cond)
            label_vec[cond_number * num_words_per_cond + word_number] = 1
            # append data
            data_X = sp.vstack((data_X, feature_vec)) if data_X is not None else feature_vec
            data_Y = np.vstack((data_Y, label_vec)) if data_Y is not None else label_vec

    # save data file
    pickle.dump(data_X, open(file_X, "wb"))
    pickle.dump(data_Y, open(file_Y, "wb"))

    return data_X, data_Y


def convert_1d_to_3d(data_X, data_Y):
    """
    Parses the 1d data set into three separate data sets for each axis.
    Each trial's data is split into [0th slice, ..., (dim-1)th slice] for each x, y, and z.
    To get only the Nth slices of an axis across all trials, traverse through data_dim_axis by index [N, N + dim-1, N + 2*dim-1, ...]

    :param data_X: whole data set
    :param data_Y: whole label set
    :return: data sets and label sets for each axis
    """

    data_dim_x = []  # slices along x-axis (has shape of (total_trials * dim_x, dim_z, dim_y))
    data_dim_x_label = []  # contains (total_trials * dim_x) labels
    data_dim_y = []  # slices along y-axis (has shape of (total_trials * dim_y, dim_z, dim_x))
    data_dim_y_label = []  # contains (total_trials * dim_y) labels
    data_dim_z = []  # slices along z-axis (has shape of (total_trials * dim_z, dim_y, dim_x))
    data_dim_z_label = []  # contains (total_trials * dim_z) labels

    for num_trial in range(data_X.shape[0]):
        label = data_Y[num_trial]
        data_1d = data_X[num_trial]
        data_3d = np.squeeze(np.asarray(data_1d.todense())).reshape((dim_z, dim_y, dim_x))
        for x in range(dim_x):
            x_slice = data_3d[:,:,x]
            # append only if the slice is not empty 
            if x_slice.sum() != 0:
                data_dim_x.append(data_3d[:, :, x])
                data_dim_x_label.append(label)
        for y in range(dim_y):
            y_slice = data_3d[:, y, :]
            if y_slice.sum() != 0:
                data_dim_y.append(data_3d[:, y, :])
                data_dim_y_label.append(label)
        for z in range(dim_z):
            z_slice = data_3d[:, :, z]
            if z_slice.sum() != 0:
                data_dim_z.append(data_3d[z, :, :])
                data_dim_z_label.append(label)

    return np.array(data_dim_x), np.array(data_dim_x_label), \
           np.array(data_dim_y), np.array(data_dim_y_label), \
           np.array(data_dim_z), np.array(data_dim_z_label)


if __name__ == "__main__":
    data_X, data_Y = load_data()
    data_dim_x, data_dim_x_label, data_dim_y, data_dim_y_label, data_dim_z, data_dim_z_label = convert_1d_to_3d(data_X, data_Y)
