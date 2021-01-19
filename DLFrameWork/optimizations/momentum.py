
import numpy as np
class Momentum:
    def __init__(self,paramterts,lr=0.1,beta=0.9):
        self.params = paramterts
        self.lr = lr
        self.beta = beta 
        self.v = {}
     
    def initialize_velocity(self):
        L = len(self.params) // 2
        v = {}
        for l in range(L):
            v['dW'+str(l+1)] = np.zeros(self.params['W'+str(l+1)].shape) 
            v['db'+str(l+1)] = np.zeros(self.params['b'+str(l+1)].shape)
        self.v = v 
    def update(self):
        L = len(self.params) // 2 
        for l in range(L):
            self.v["dW" + str(l+1)] = (beta*self.v["dW" + str(l+1)]) + ((1-beta)*self.params['W'+str(l+1)])
            self.v["db" + str(l+1)] = (beta*self.v["db" + str(l+1)]) + ((1-beta)*self.params['b'+str(l+1)])
            self.params["W" + str(l+1)] = self.params["W" + str(l+1)] - (self.lr*self.v["dW" + str(l+1)])
            self.params["b" + str(l+1)] = self.params["b" + str(l+1)] - (self.lr*self.v["db" + str(l+1)])
    def __repr__(self):
        self.initialize_velocity()
        self.update()
        return {'velocity':self.v,
                    'parameters':self.params}    

                
