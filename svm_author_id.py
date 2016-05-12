#!/usr/bin/python

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


from sklearn import svm
clf = svm.SVC(kernel="rbf",C=10000.0)
#features_train = features_train[:len(features_train)/100]
#labels_train = labels_train[:len(labels_train)/100]
t0 = time()
print "fit begin"
fit = clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3), "s"

t1 = time()
pred = clf.predict(features_test)
print "predicting time:", round(time()-t1, 3), "s"
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred, labels_test)
print(accuracy)

a, b = 0
for item in pred:
	if item == 1:
		a += 1
	else:
		b += 1
print "Chris predicted msg :", a
print "Sara  predicted msg :", b

