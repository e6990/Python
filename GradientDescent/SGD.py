# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 21:11:04 2017

@author: e6990
"""

import numpy as np
from numpy.random import seed

class GradientDescent(object):
    def __init__(self,step,n_iter,shuffle = True, random_state = None):
        self.step = step;
        self.n_iter = n_iter;
        self.shuffle = shuffle;
        if random_state:
            seed(random_state);
            
    def fix(self,X,Y):
        self.w_ = np.zeros(X.shape[1]+1);
        self.cost = [];
        for i in range(self.n_iter):
            if self.shuffle:
                r = np.random.permutation(len(Y));
                X = X[r];
                Y = Y[r];
            cost = [];
            for xi,target in zip(X,Y):
                xi = np.asarray(xi);
                target = np.asarray(target);
                output = np.dot(xi,self.w_[1:])+self.w_[0];
                error = target - output;
                self.w_[1:] += self.step * np.dot(xi,error);
                self.w_[0] += self.step * error.sum();
                cost.append(0.5*error**2);
            avg_cost = sum(cost)/len(Y);
            self.cost.append(avg_cost);
        return self;
    
    def predict(self,X):
        return np.where(np.dot(X,self.w_[1:]) + self.w_[0] > 0,1,-1);