#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from sklearn.naive_bayes import GaussianNB

sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################

# create Naive Bayes classifier
clf = GaussianNB(priors=None)

# train the classifier using training data
t0 = time()
clf.fit(features_train, labels_train)
print('training time: {} s'.format(round(time()-t0, 3)))

# predict labels for test data
t0 = time()
labels_predict = clf.predict(features_test)
print('prediction time: {} s'.format(round(time()-t0, 3)))

# get the mean accuracy of the classifier based on test data
score = clf.score(features_test, labels_test)
print('mean accuracy: {:.3f}'.format(score))

#########################################################
