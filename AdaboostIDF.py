import sys
# I need this because my python installation is weird..
sys.path.append('/usr/local/lib/python2.7/site-packages')

from sklearn import tree
from sklearn import ensemble
import csv
import numpy as np
import matplotlib.pyplot as plt

# cross validation
from sklearn import cross_validation

from linRegClassifier import linRegClassifier

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

# Used for cross validation to select parameters
def get_error(G, Y):
    error = 0
    for i in range(len(G)):
        if G[i] != Y[i]:
            error += 1
    return 1.0 * error / len(G)


#min_samples_leafs = [i for i in range(10, 30)]
# NOTE: Just decided 12 here from looking at graphs during cross validation.
# Change back to previous line if you want to see the range
#min_samples_leafs = [18]
# 22 is best right now: 0.894982676926
# 27 gives avg = 0.885996103119
test_errors = []
train_errors = []

n_estimators = [500]

for n in n_estimators:
    print "Working on n... = ", n
    # initialize the tree model 
    clf = ensemble.AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(max_depth=1), n_estimators = n)
    #clf = ensemble.AdaBoostClassifier(base_estimator = linRegClassifier(), n_estimators = n, algorithm='SAMME', learning_rate=0.001)
    # train the model

    # DONT NEED THIS ATM
    clf = clf.fit(X_train, Y_train)

    # make prediction
    G_train = clf.predict(X_train)
    G_test = clf.predict(X_test)
    G_testFile = clf.predict(X_testFile)
    #print G_testFile

    # compute error 

    K = 5
    scores = cross_validation.cross_val_score(clf, X_train, Y_train, cv=K, scoring='accuracy', verbose = 1, n_jobs = -1)
    avg_score = sum(scores) / len(scores)
    print('Scores = {}'.format(scores))
    print('avg_score = {}'.format(avg_score))

    # NOTE: Uncomment if doing cross val
    #train_error = get_error(G_train, Y_train)
    #train_errors.append(train_error)
    #test_error = get_error(G_test, Y_test)
    #test_errors.append(test_error)

"""
plt.plot(min_samples_leafs, train_errors)
plt.plot(min_samples_leafs, test_errors)
plt.xlabel('min_samples_leaf')
plt.ylabel('Error')
plt.title('Plot of Error vs. min_samples_leaf')
plt.legend(['train_error', 'test_error'])
plt.show()
"""


"""
f = open('predictionsIDF-Adaboost400.csv','w')
f.write('Id,Prediction\n')
for (i, e) in enumerate(G_testFile):
    #print i, e
    f.write('%d,%d\n' % (i+1, e))
"""

