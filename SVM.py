import sys
# I need this because my python installation is weird..
#sys.path.append('/usr/local/lib/python2.7/site-packages')

from sklearn import tree
from sklearn import ensemble
import csv
import numpy as np
import matplotlib.pyplot as plt

# cross validation
from sklearn import cross_validation
from sklearn import svm

def get_error(G, Y):
    error = 0
    for i in range(len(G)):
        if G[i] != Y[i]:
            error += 1
    return 1.0 * error / len(G)

# NOTE: Decrease if you want to do some cross validation. 
# (just changed to 4000 to train the final model, after selected leaf
# parameter via cross valiation)
NUM_TRAININGS = 4000

fin_name = 'kaggle_train_tf_idf.csv'
fout_name = 'kaggle_test_tf_idf.csv'

with open(fin_name, 'r') as fin:
    next(fin)
    data = np.array(list(csv.reader(fin))).astype(float)

X_train = data[:NUM_TRAININGS, 1:-1]
Y_train = data[:NUM_TRAININGS, -1]

# these will be empty unless you do some cross validation
X_test = data[NUM_TRAININGS:, 1:-1]
Y_test = data[NUM_TRAININGS:, -1]

# grab the real test data
with open(fout_name, 'r') as fout:
    next(fout)
    data = np.array(list(csv.reader(fout))).astype(float)

X_testFile = data[:, 1:]
#Y_testFile = data[:, -1] # Note: theres no Y predictions for the real test data :)

for C in [100, 250, 500, 750, 1000, 2000, 3000, 10000, 11000, 12000, 13000, 20000]:
    '''
    clf = svm.SVC(C=1.0, kernel='poly', degree=3, gamma=0.0, coef0=coeff,
                  shrinking=True, probability=False, tol=0.001, cache_size=200,
                  class_weight=None, verbose=False, max_iter=-1, random_state=None)
    '''
    
    clf = svm.LinearSVC(penalty='l1', loss='l2', dual=False, tol=0.0001, C=C,
                        multi_class='ovr', fit_intercept=True, intercept_scaling=1,
                        class_weight=None, verbose=0, random_state=None)
    
    '''
    clf = svm.NuSVC(nu=nu, kernel='linear', degree=3, gamma=0.0, coef0=0.0,
                    shrinking=False, probability=False, tol=0.001, cache_size=200,
                    verbose=False, max_iter=-1, random_state=None)
    '''
    
    #clf.fit(X_train, Y_train) 
    #G_train = clf.predict(X_train)
    
    K = 5
    scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=K, scoring='accuracy', verbose = 0, n_jobs = -1)
    print('Scores = {}'.format(scores))
    
'''
clf.fit(X_train, Y_train)
G_testFile = clf.predict(X_testFile)
f = open('predictionsSvm.csv','w')
f.write('Id,Prediction\n')
for (i, e) in enumerate(G_testFile):
    #print i, e
    f.write('%d,%d\n' % (i+1, e))
'''