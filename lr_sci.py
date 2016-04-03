#lr using scikit api 

import numpy as np
from scipy import sparse as sp
import pickle
from sklearn import linear_model

# import some data to play with
fx = open('x_sparse.p')
fy = open('y.p')

X = pickle.load(fx)
Y = pickle.load(fy)

temp = np.array([range(0,60)])
Y = (Y.dot(temp.T)).ravel()

h = .02  # step size

#logreg = linear_model.LogisticRegression(penalty='l2', solver='sag', verbose=1)
logreg = linear_model.LogisticRegression(penalty='l2', solver='newton_cg', verbose=1)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)
