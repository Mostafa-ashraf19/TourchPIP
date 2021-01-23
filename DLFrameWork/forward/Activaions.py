import numpy as np
from .Layers import Layer_Dense

class Sigmoid:
    def forwards(self, inputs):
        X = inputs.out
        sig =   1 / (1 + (np.exp(-1 * X)))
        inputs.pass_act('Sigmoid',sig)
        return sig
    @staticmethod
    def sigmoid_(inputs):
        return 1 / (1 + (np.exp(-inputs)))
        # pass
    @staticmethod    
    def sigmoidBW_(inputs):
        return Sigmoid.sigmoid_(inputs)* (1-Sigmoid.sigmoid_(inputs))    

    def backwards(self, inputs):
        s = self.forwards(inputs)
        return s * (1 - s)

class ReLU:

    def forwards(self, inputs):
        rel = np.maximum(0, inputs.out)
        inputs.pass_act('relu',rel)
        return rel
    @staticmethod     
    def ReLU_(inputs):
        # print('relu in value is', inputs)
        # print('-'*20)
        # print('relu out value is',np.maximum(0, inputs))
        return np.maximum(0, inputs)    
    @staticmethod
    def ReLUBW_(inputs):
        Z = inputs
        dZ = np.array(inputs, copy=True) # just converting dz to a correct object.
        # When z <= 0, you should set dz to 0 as well. 
        dZ[Z <= 0] = 0
        return dZ

        # return np.maximum(0, inputs)#1 if inputs > 0 else 0

    def Backwards(self, inputs):
         if inputs > 0:
             return 1
         elif inputs <= 0:
             return 0

class Identity:
    def forwards(self, inputs):
        ident =inputs.out
        inputs.pass_act('identity',ident)
        return ident

    def backwards(self, inputs):
        pass

class Tanh:
    def forwards(self, inputs):
        tan = np.tanh(inputs.out)
        inputs.pass_act('tanh',tan)
        return tan
    @staticmethod
    def Tanh_(inputs):
        return np.tanh(inputs)
    @staticmethod 
    def TanhBW_(inputs):
        return (1 - Tanh.Tanh_(inputs)**2)

    def backwards(self, inputs):
        a = self.forwards(inputs)
        return 1 - a ** 2


class Softmax:
    @staticmethod
    def forward(inputs):
        exp_x = np.exp(inputs.out)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        # inputs.pass_act('softmax',probs)
        return probs
    @staticmethod    
    def softmax_grad(s): 
        jacobian_m = np.diag(s)    
        for i in range(len(jacobian_m)):
            for j in range(len(jacobian_m)):
                if i == j:
                    jacobian_m[i][j] = s[i] * (1-s[i])
                else: 
                    jacobian_m[i][j] = -s[i]*s[j]
        return jacobian_m    

