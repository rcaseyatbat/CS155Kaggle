import sys
# I need this because my python installation is weird..
sys.path.append('/usr/local/lib/python2.7/site-packages')

from sklearn import tree
from sklearn import ensemble
from sklearn.base import BaseEstimator, ClassifierMixin

import csv
import numpy as np
import matplotlib.pyplot as plt
import random

NUM_FEATURES = 500
reg = 0.0 # Regularization term

'''
A classifier which classifies based on the dot product of the features of
articles and the hidden features of the article types [0, 1].
'''
class linRegClassifier(BaseEstimator, ClassifierMixin):
    
    def __init__(self):
        self.classes_ = np.unique([0, 1])
        self.W = np.zeros(NUM_FEATURES)        
        pass
    
    def fit(self, X, y, sample_weight=None):    
        X_train = np.matrix(np.copy(X))
        Y_train = np.copy(y)
        # Given a weighted sample, multiply the correspond Y and X values by
        # that value
        if sample_weight is not None:
            num_samples = len(sample_weight)
            Y_train = Y_train * sample_weight * num_samples
            for i in range(num_samples):
                X_train[i, :] = X_train[i, :] * sample_weight[i] * num_samples
        
        # Minimize with the pseudo inverse
        self.W = np.array(Y_train*X_train*np.matrix.getI(np.matrix.getT(X_train) * X_train + reg * np.identity(NUM_FEATURES)))
        return self
    
    # Returns an array of predictions for the num_samples x num_dim matrix X
    def predict(self, X):
        X = np.matrix(X)
        num_samples = len(X[:, 0])
        y = np.zeros(num_samples)
        for i in range(num_samples):
            dot = np.dot(self.W, np.transpose(X[i,:]))[0, 0]
            if dot < 0.5:
                y[i] = 0
            else:
                y[i] = 1
        return y