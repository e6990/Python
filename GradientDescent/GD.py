# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 22:51:26 2017

@author: e6990
"""

import numpy as np

class GradientDescent(object):
    def __init__(self,step ,n_iter):
        self.step = step;
        self.n_iter = n_iter;
        
    def fix(self,X,Y):
        '行向量为样本，列向量为属性'
        X = np.asarray(X);
        '''标准化，便于统一数据'''

        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-X[:,i].mean()) / X.std();
            
        Y = np.asarray(Y);
        self.w_ = np.zeros(X.shape[1] + 1);
        self.error = [];
        for _ in range(self.n_iter):
            '''
            df/dw_ = -lim(y-wx)*x
            loss = 1/2 * lim(y - wx)^2
            '''
            subvalue = Y - (np.dot(X,self.w_[1:]) + self.w_[0]);
            self.w_[1:] += self.step * X.T.dot(subvalue);
            self.w_[0] += self.step * subvalue.sum();
            error =(subvalue ** 2).sum() / 2.0;
            self.error.append(error);
        return self;
    
    def predict(self,X):
        return np.where(np.dot(X,self.w_[1:]) + self.w_[0] > 0,1,-1);