import sys
# I need this because my python installation is weird..
sys.path.append('/usr/local/lib/python2.7/site-packages')

from sklearn import tree
import csv
import numpy as np
import matplotlib.pyplot as plt


# NOTE: Decrease if you want to do some cross validation. 
# (just changed to 4000 to train the final model, after selected leaf
# parameter via cross valiation)
NUM_TRAININGS = 4000

fin_name = 'kaggle_train_wc.csv'
fout_name = 'kaggle_test_wc.csv'

with open(fin_name, 'r') as fin:
    next(fin)
    data = np.array(list(csv.reader(fin))).astype(int)

X_train = data[:NUM_TRAININGS, 1:-1]
Y_train = data[:NUM_TRAININGS, -1]

# these will be empty unless you do some cross validation
X_test = data[NUM_TRAININGS:, 1:-1]
Y_test = data[NUM_TRAININGS:, -1]

# grab the real test data
with open(fout_name, 'r') as fout:
    next(fout)
    data = np.array(list(csv.reader(fout))).astype(int)

X_testFile = data[:, 1:]
#Y_testFile = data[:, -1] # Note: theres no Y predictions for the real test data :)

# Used for cross validation to select parameters
def get_error(G, Y):
    error = 0
    for i in range(len(G)):
        if G[i] != Y[i]:
            error += 1
    return 1.0 * error / len(G)


#min_samples_leafs = [i for i in range(1, 25)]
# NOTE: Just decided 12 here from looking at graphs during cross validation.
# Change back to previous line if you want to see the range
min_samples_leafs = [12]
test_errors = []
train_errors = []

for min_samples_leaf in min_samples_leafs:
    # initialize the tree model 
    clf = tree.DecisionTreeClassifier(criterion='gini', 
        min_samples_leaf=min_samples_leaf)
    # train the model
    clf = clf.fit(X_train, Y_train)

    # make prediction
    G_train = clf.predict(X_train)
    G_test = clf.predict(X_test)
    G_testFile = clf.predict(X_testFile)
    print G_testFile

    # compute error 

    # NOTE: Uncomment if doing gross val
    #train_error = get_error(G_train, Y_train)
    #train_errors.append(train_error)
    #test_error = get_error(G_test, Y_test)
    #test_errors.append(test_error)


f = open('predictions.csv','w')
f.write('Id,Prediction\n')
for (i, e) in enumerate(G_testFile):
    #print i, e
    f.write('%d,%d\n' % (i+1, e))


