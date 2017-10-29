# -*- coding: utf-8- -*-

import numpy as np
class perceptron(object):
    def __init__(self,step = 0.01,n_iter = 10):
        self.step = step;
        self.n_iter = n_iter;
    
    def predict(self,x):
        forecast = np.dot(x,self.w_[1:]) + self.w_[0];
        return np.where(forecast > 0.0, 1,-1);
    
    def fix(self,x,y):
        #x行为样本，列为属性
        x = np.asarray(x);
        y = np.asarray(y);
        #初始化权重 w+1 ,多出一个为偏移量值
        self.w_ = np.zeros(1 + x.shape[1]);
        self.error = [];
        self.ws = [];
        for _ in range(self.n_iter):
            error = 0;
            #Python赋值是指向内存所在空间，空间内值改变，赋值对象值改变
            w = self.w_;
            w = tuple(w);
            for sample,target in zip(x,y):
                update = self.step * (target - self.predict(sample));
                #w比x维度多1,乘以sample因为wx关于w的微分为x
                self.w_[1:] +=  update * sample;
                self.w_[0] += update;
                error += (int)(update != 0.0);
            self.ws.append(w);
            self.error.append(error);
        return self;