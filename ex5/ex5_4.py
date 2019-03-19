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
import logging
logger = logging.getLogger()
logger.disabled = True

digits = load_digits()

x_train, x_test, y_train, y_test = train_test_split(
 digits.data, digits.target, test_size=0.20, train_size=0.80)

transformer= Normalizer().fit(x_train)
X_train=transformer.transform(x_train)
transformer= Normalizer().fit(x_test)
X_test=transformer.transform(x_test)

C_range=np.linspace(10 ** -5,1,10)
clf_list = [LogisticRegression(), SVC()]
clf_name = ['LR', 'SVC']

for clf,name in zip(clf_list, clf_name):
    for C in C_range:
        for penalty in ["l1", "l2"]:
            clf.C = C
            clf.penalty = penalty
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            print("The accuracy score of"+name+", penalty="+clf.penalty+" C="+str(clf.C)+"is" + str(score))

#The accuracy score ofSVC, penalty=l2 C=1.0 is 0.6833333333333333
#The accuracy score ofSVC, penalty=l2 or 11 C=1.0 is 0.6833333333333333
