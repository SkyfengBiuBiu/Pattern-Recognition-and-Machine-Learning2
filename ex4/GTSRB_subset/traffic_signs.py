# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 08:48:51 2019

@author: hehu
"""

import glob
import numpy as np
import os
import matplotlib.pyplot as plt
from simplelbp import local_binary_pattern

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,GradientBoostingClassifier



def load_data(folder):
    """ 
    Load all images from subdirectories of
    'folder'. The subdirectory name indicates
    the class.
    """
    
    X = []          # Images go here
    y = []          # Class labels go here
    classes = []    # All class names go here
    
    subdirectories = glob.glob(folder + "/*")
    
    # Loop over all folders
    for d in subdirectories:
        
        # Find all files from this folder
        files = glob.glob(d + os.sep + "*.jpg")
        
        # Load all files
        for name in files:
            
            # Load image and parse class name
            img = plt.imread(name)
            class_name = name.split(os.sep)[-2]

            # Convert class names to integer indices:
            if class_name not in classes:
                classes.append(class_name)
            
            class_idx = classes.index(class_name)
            
            X.append(img)
            y.append(class_idx)
    
    # Convert python lists to contiguous numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def extract_lbp_features(X, P = 8, R = 5):
    """
    Extract LBP features from all input samples.
    - R is radius parameter
    - P is the number of angles for LBP
    """
    
    F = [] # Features are stored here
    
    N = X.shape[0]
    for k in range(N):
        
        print("Processing image {}/{}".format(k+1, N))
        
        image = X[k, ...]
        lbp = local_binary_pattern(image, P, R)
        hist = np.histogram(lbp, bins=range(257))[0]
        F.append(hist)

    return np.array(F)

# Test our loader

X, y = load_data(".")
F = extract_lbp_features(X)
print("X shape: " + str(X.shape))
print("F shape: " + str(F.shape))

# Continue your code here...
#%%
#ex4_4
model1=KNeighborsClassifier()
model2=LinearDiscriminantAnalysis()
model3=SVC()
model4=LogisticRegression()
models=[model1,model2,model3,model4]

for model in models:
    print("The accuracy score is"+str(cross_val_score(model, F, y)))
    
#%%
clf = RandomForestClassifier(n_estimators=100)
print("The accuracy score is"+str(cross_val_score(clf, F, y)))

clf = ExtraTreesClassifier(n_estimators=100)
print("The accuracy score is"+str(cross_val_score(clf, F, y)))

clf = GradientBoostingClassifier(n_estimators=100)
print("The accuracy score is"+str(cross_val_score(clf, F, y)))

clf = AdaBoostClassifier(n_estimators=100)
print("The accuracy score is"+str(cross_val_score(clf, F, y)))