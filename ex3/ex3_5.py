# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 13:35:15 2019

@author: ThinkPad
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier  
import sklearn.metrics as sk
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split


mat = loadmat("./Ex1_data/twoClassData.mat")
X = mat["X"]
y = mat["y"].ravel()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)
classifier = KNeighborsClassifier()  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  
print(sk.accuracy_score(y_test, y_pred))  


#LDA
clf = LinearDiscriminantAnalysis()
clf.fit(X_train, y_train)
y_pre=clf.predict(X_test)
print(sk.accuracy_score(y_test, y_pre))  