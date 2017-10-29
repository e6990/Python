# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 17:46:39 2017

@author: e6990
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import GD
import SGD
from matplotlib.colors import ListedColormap

def decision_regions(X,Y,classifier,step = 0.02):
    X = np.asarray(X);
    Y = np.asarray(Y);
    markers = ('s','x','o','^','v');
    colors = ('red','blue','lightgreen','gray','cyan');
    cmap = ListedColormap(colors[:len(np.unique(Y))]);
    #plot the decision surface
    x1_min,x1_max = X[:,0].min() - 1, X[:,0].max() + 1;
    x2_min,x2_max = X[:,1].min() - 1, X[:,1].max() + 1;
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,step),np.arange(x2_min,x2_max,step));
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T);
    Z = Z.reshape(xx1.shape);
    plt.contourf(xx1, xx2, Z, alpha = 0.5, cmap = cmap);
    plt.xlim(x1_min,x1_max);
    plt.ylim(x2_min,x2_max);
    
    #plot the class smaple
    for idx,sample in enumerate(np.unique(Y)):
        plt.scatter(X[Y == sample,0],X[Y == sample,1],alpha = 0.8, color = cmap(idx),marker = markers[idx],label = idx);
    

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header = None);
y = df.iloc[0:100,4].values;
y = np.where(y == 'Iris-setosa',1,-1);
x = df.iloc[0:100,[0,2]].values;
positive = np.where(y == 1);
negative = np.where(y == -1);
plt.scatter(x[positive,0], x[positive,1], color = 'red', marker = 'o', label = 'setosa');
plt.scatter(x[negative,0], x[negative,1], color = 'blue', marker = '*', label = 'versicolor');
plt.xlabel('petal length');
plt.ylabel('sepal length');
plt.legend(loc='upper left');
plt.show();
npp = SGD.GradientDescent(step = 0.01,n_iter = 50);
npp.fix(x,y);
decision_regions(x,y,classifier = npp);
'''
plt.scatter(range(1,len(npp.error)+1),npp.error,color = 'green', marker = 'o');
plt.xlabel('number of sample');
plt.ylabel('error');
plt.show();
'''
