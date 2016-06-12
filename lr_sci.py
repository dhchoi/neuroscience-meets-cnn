#lr using scikit api 

import numpy as np
from scipy import sparse as sp
import pickle
from sklearn import linear_model

# import some data to play with
#fx = open('x_sparse.p')
#fx = open('x_pca.p')
fx = open('x_lda.p')
fy = open('y.p')

X = pickle.load(fx)
Y = pickle.load(fy)

temp = np.array([range(0,60)])
Y = (Y.dot(temp.T)).ravel()

num_subjects = 9
tot = 0
for i in range(num_subjects):
    print "=== using subject", i+1, "as test set"
    logreg = linear_model.LogisticRegression(penalty='l2', solver='newton-cg', n_jobs = -1, verbose=1)
    X = np.concatenate((X[360:], X[:360]), axis=0)
    Y = np.concatenate((Y[360:], Y[:360]), axis=0)
    logreg.fit(X[:X.shape[0]-360,:], Y[:Y.shape[0]-360])
    acc = sum(logreg.predict(X[3240-360:,:])==Y[3240-360:]) / 360.0
    print "accuracy : ", acc
    tot += acc

print "mean accuracy : ", tot / (1.0 * num_subjects)
