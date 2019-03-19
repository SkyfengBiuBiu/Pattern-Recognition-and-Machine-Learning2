# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 15:31:11 2019

@author: ThinkPad
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

mat=loadmat('E:\\Pattern Recognition and Machine Learning\\ex7\\arcene\\arcene.mat')
X_train=mat["X_train"]
X_test=mat["X_test"]
y_train=mat["y_train"]
y_test=mat["y_test"]

estimator = SVR(kernel="linear")
rfe = RFECV(estimator,step=50, cv=5,verbose=1)
rfe = rfe.fit(X_train,np.asarray(y_train).transpose().ravel() )
print(rfe.support_)
plt.plot(range(0,10001,50), rfe.grid_scores_)
plt.show()
y_pred = rfe.predict(X_test)
score = accuracy_score(np.asarray(y_test).transpose().ravel(),y_pred.round() )
print(score)

#%%
#ex7_5

C_range=np.linspace(10 ** -5,1,10)
clf=LogisticRegression()
gre=0
C_gr=0
for C in C_range:
    clf.C = C
    clf.penalty = "l1"
    score1 = cross_val_score(clf,X_train, np.asarray(y_train).transpose().ravel(),cv=10)
    print("The accuracy score of" + str(clf.C) + "is" + str(score1))
    current=np.mean(score1)
    if(current>gre):
        gre=current
        C_gr=C


#%%
clf.C = C_gr
clf.fit(X_train, np.asarray(y_train).transpose().ravel())
b=np.nonzero(clf.coef_)
print("Count the number of selected features"+str(len(b)))
y_pred = clf.predict(X_test)
score2 = accuracy_score(np.asarray(y_test).transpose().ravel(),y_pred.round() )
print("The accuracy on X_testandy_test"+ str(score2))