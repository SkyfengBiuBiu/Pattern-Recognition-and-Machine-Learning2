# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:19:10 2019

@author: ThinkPad
"""

import winsound
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal


n1=np.zeros(500)
n=np.arange(500,600)
n2=np.cos(2*np.pi*0.1*n)
n3=np.zeros(300)
y=np.concatenate((n1,n2,n3))

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(np.arange(0,900), y)
plt.title('Noiseless signal')
plt.axis('tight')
plt.grid('on')

#Add noises
y_n = y + np.sqrt(0.5)*np.random.randn(y.size)

plt.subplot(3, 1, 2)
plt.plot(np.arange(0,900), y_n)
plt.title('Noisy signal')
plt.axis('tight')
plt.grid('on')

#Implement the sinal detector for known signal version
h=np.cos(2*np.pi*0.1*n)
y_d=np.convolve(h,y_n,'same')
plt.subplot(3, 1, 3)
plt.plot(np.arange(0,900), y_d)
plt.title('Detection random signal')
plt.axis('tight')
plt.grid('on')
