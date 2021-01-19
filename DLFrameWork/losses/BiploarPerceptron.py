import numpy as np
from ..forward import Layer_Dense


class BiploarPerceptron(Layer_Dense):
    def __init__(self,pred,lable):
        self.pred = pred
        self.lable = lable
        self.parameters = Layer_Dense._params #parameter
        self.dw={}
    def loss(self):
        return max(0,-(self.lable+self.parameters['Z'+str(len(Layer_Dense.layer_activations.keys()))]))
    def preceptron_loss_grad(self):
        dwm =[]
        L =len(self.lable)
        for l in reversed(range(L)): 
            if (self.lable[l]*np.dot(self.parameters['w'+str(l)].shap,self.data[l])) > 0:
              dwm[l]= np.zeros(self.parameters['w'+str(l)].shape)
            else:
              for j in range (0,L):
                  dwm[j] = np.dot(self.lable[l],self.data[j])  
            self.dw['dw'+str(l)]= dwm    