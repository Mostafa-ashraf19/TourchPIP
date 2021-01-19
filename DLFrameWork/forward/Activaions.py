import numpy as np
from Layers import Layer_Dense

class Sigmoid:
    
    def forwards(self, inputs):
        X = inputs.out
        sig =   1 / (1 + (np.exp(-1 * X)))
        inputs.pass_act('sigmoid',sig)
        return sig

    def backwards(self, inputs):
        s = self.forwards(inputs)
        return s * (1 - s)

class ReLU:

    def forwards(self, inputs):
        rel = np.maximum(0, inputs.out)
        inputs.pass_act('relu',rel)
        return rel 

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


    def backwards(self, inputs):
        a = self.forwards(inputs)
        return 1 - a ** 2
class Softmax:
    def forward(self, inputs):
        exp_x = np.exp(inputs.out)
        probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        inputs.pass_act('softmax',probs)
        return probs