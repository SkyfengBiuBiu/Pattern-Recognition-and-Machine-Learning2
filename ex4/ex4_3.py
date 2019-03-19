# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 20:28:28 2019

@author: ThinkPad
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

digits = load_digits()
print(digits.keys())


plt.gray()
plt.imshow(digits.images[0])
plt.show()
print(digits.target[0])
print(digits.data[0].size)

x_train, x_test, y_train, y_test = train_test_split(
 digits.data, digits.target, test_size=0.20, train_size=0.80)

model1=KNeighborsClassifier()
model2=LinearDiscriminantAnalysis()
model3=SVC()
model4=LogisticRegression()
models=[model1,model2,model3,model4]

for model in models:
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    print("The accuracy score is"+str(accuracy_score(y_test, y_pred)))