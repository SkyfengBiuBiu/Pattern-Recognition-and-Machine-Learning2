# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 20:28:28 2019

@author: ThinkPad
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(
 digits.data, digits.target, test_size=0.20, train_size=0.80)

transformer= Normalizer().fit(x_train)
X_train=transformer.transform(x_train)
transformer= Normalizer().fit(x_test)
X_test=transformer.transform(x_test)

clf_list = [LogisticRegression(), SVC()]
clf_name = ['LR', 'SVC']
penalty = ['l1', 'l2']


clf=clf_list[0]
hyperparameters = dict()
clf = RandomizedSearchCV(clf, hyperparameters, cv=5, verbose=0)
clf.fit(X_train, y_train)
best_model = clf.fit(X_train, y_train)
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

clf=clf_list[1]
clf =  RandomizedSearchCV(clf,hyperparameters, cv=5)
clf.fit(X_train, y_train)
best_model = clf.fit(X_train, y_train)
print('Best C:', best_model.best_estimator_.get_params()['C'])