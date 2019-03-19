# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 15:10:54 2019

@author: ThinkPad
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat



if __name__=="__main__":
    
    #The first question
    X=[]
    X=np.loadtxt("./locationData/locationData.csv")
    print("all data read.")
    print("Result size is %s" %(str(X.shape)))
    
    #The second question
    plt.subplot(4, 1, 1)
    plt.plot(X[:,0],X[:,1])
    ax = plt.subplot(4, 1, 2, projection = "3d")
    plt.plot(X[:,0], X[:,1], X[:,2])
    
    #The third question
    X1=[]
    with open("./locationData/locationData.csv","r") as fp:
        for line in fp:
            values = line.split(";")
            values = str(values).split()
            
            values[0]=values[0].replace('[\'','')
            values[2]=values[2].replace('\\n\']','')
            values = [float(v) for v in values]
            #print(values)
            X1.append(values)
            #print(X)
    X1=np.array(X1)
    print("Result size is %s" %(str(X1.shape)))
    print((X==X1).all())
    
    #The fourth question
    mat = loadmat("./Ex1_data/twoClassData.mat")
    print(mat.keys())
    X = mat["X"]
    y = mat["y"].ravel()
    class0=X[y == 0, :]
    class1=X[y == 1, :]
    
    plt.subplot(4, 1, 3)
    plt.plot(class0[:, 0], class0[:, 1], 'ro')
    plt.plot(class1[:,0], class1[:, 1], 'bo')
    
    #The fifthth question
    x=np.load("./least_squares_data/x.npy")
    y=np.load("./least_squares_data/y.npy")
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.subplot(4, 1, 4)
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, m*x + c, 'r', label='Fitted line')
    plt.legend()
    plt.show()