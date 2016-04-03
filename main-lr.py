import pickle
import numpy as np
import scipy.io as sio

num_subjects = 9
num_words_per_cond = 5
num_conds = 12
dim_x = 51
dim_y = 61
dim_z = 23

file_X = "x.p"
file_Y = "y.p"


def load_data(from_stored_data=False):
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
                x = colInfo[0]
                y = colInfo[1]
                z = colInfo[2]
                feature_vec[z * (dim_x * dim_y) + y * dim_x + x] = voxels[i]
            # create label vectors
            trial_info = info[num_trial]
            cond_number = trial_info[1][0][0] - 2  # starts from 2 (2 ~ 13)
            word_number = trial_info[3][0][0] - 1  # starts from 1 (1 ~ 5)
            label_vec = np.zeros(num_conds * num_words_per_cond)
            label_vec[cond_number * num_words_per_cond + word_number] = 1
            # append data
            data_X = np.vstack((data_X, feature_vec)) if data_X is not None else feature_vec
            data_Y = np.vstack((data_Y, label_vec)) if data_Y is not None else label_vec

    # save data file
    pickle.dump(data_X, open(file_X, "wb"))
    pickle.dump(data_Y, open(file_Y, "wb"))

    return data_X, data_Y


if __name__ == "__main__":
    data_X, data_Y = load_data()
