import numpy as np
import pickle
import time 
import scipy.sparse as sp

def preprocess(data, label):
    #pdb.set_trace()
    res_data = np.zeros((data.shape[0]*100, 21, 57, 20))
    res_label = np.zeros((data.shape[0]*100, 60))
    print data.shape
    for i in range(data.shape[0]):
        ith = data[i,:,:,:]
        for x in range(5):
            for y in range(5):
                for z in range(4):
                    ind = z + 4*y + 20*x + 100*i
                    res_data[ind] = ith[x:x+21, y:y+57, z:z+20]
                    res_label[ind] = label[i]
    return res_data, res_label


if __name__ == "__main__":
    data = pickle.load(open('data/ind_1_x', 'rb'))
    label = pickle.load(open('data/ind_1_y', 'rb'))

    data = data.todense()

    dim_x_half = 25
    dim_y = 61
    dim_z = 23
    data_3d = []  # data_3d.shape = (num_data, dim_x, dim_y, dim_z)
    data_3d = []  # data_3d.shape = (num_data, dim_x_half, dim_y, dim_z)

    for i in range(len(data)):
        d_3d = np.squeeze(np.asarray(data[i])).reshape((dim_x_half, dim_y, dim_z))
        data_3d.append(d_3d)
    data_3d = np.array(data_3d)

    print 'init done'
    t1 = time.time()
    a,b = preprocess(data_3d,label)
    t2 = time.time()

    print '%0.3f' % ((t2-t1))
    a = np.reshape(a, (a.shape[0], a.shape[1] * a.shape[2] * a.shape[3]))
    a = sp.csr_matrix(a)
    pickle.dump((a, b), open('augmented_1', 'wb'))
    
