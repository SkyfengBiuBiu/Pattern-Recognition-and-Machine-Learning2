# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 15:36:57 2019

@author: ThinkPad
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.io import loadmat

#ex2
#%%
n=np.arange(0,100,1)
f0=0.017
w = np.sqrt(0.25)*np.random.randn(100)
x=np.sin(2*np.pi*f0*n)+w
x=np.array(x)
plt.plot(n,x)

#%%
scores = []
frequencies = []

for f in np.linspace(0, 0.5, 1000):

    n = np.arange(100)
    z =(-2*(np.pi)*f*n)*(1j)
    e = np.exp(z)
    score = np.abs(np.dot(x,e))

    scores.append(score)
    frequencies.append(f)

fHat = frequencies[np.argmax(scores)]
print(fHat)