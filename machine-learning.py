from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.dummy import DummyClassifier
from scipy.io import mmread

import numpy as np

def evaluate(clf, X, y, X_test,y_test):
    clf.fit(X, y)
    y_pred = clf.predict(X_test)
    print("CA: {}\nmicro F1: {}\nmacro F1: {}".format(accuracy_score(y_test, y_pred),
                                                      f1_score(y_test, y_pred, average='micro'),
                                                      f1_score(y_test, y_pred, average='macro')))
X, X_test = np.load('data/train.npy'), np.load('data/test.npy')
y, y_test = X[:, -1], X_test[:, -1]
X, X_test = X[:,:-1], X_test[:,:-1]


print("Baseline:")
base = DummyClassifier(strategy='most_frequent')
evaluate(base, X, y, X_test, y_test)

print()
print("Multinominal logistic regression:")
clf = LogisticRegression(solver='lbfgs', multi_class='multinomial')
evaluate(clf, X, y, X_test, y_test)


#Best validation results
#CA: 0.5215716486902927
#micro F1: 0.5215716486902927
#macro F1: 0.347493580823747

#Test results
#CA: 0.583076923076923
#micro F1: 0.583076923076923
#macro F1: 0.43439536272224927