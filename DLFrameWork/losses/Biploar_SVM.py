import numpy as np
from ..forward import Layer_Dense


class Biploar_SVM(Layer_Dense) :
    def __init__(self,pred,lable):
        self.pred = pred
        self.lable = lable
        self.parameters = Layer_Dense._params #parameter
        self.dw={}
   
    def loss(self):
        return max(0,1-(self.lable+self.parameters['Z'+str(len(Layer_Dense.layer_activations.keys()))]))
        
    def SVM_loss_grad(self):
        dwm =[]
        L = len(self.lable)
        for l in reversed(range(L)):
            
            if (self.lable[l]*np.dot(self.parameters['W'+str(l)].shap,self.data[l])) > 1:
              dwm[l]= np.zeros(self.parameters['W'+str(l)].shape)
            
            else :
              for j in range (0,L):
                  dwm[j] = np.dot(self.lable[l],self.data[j])
            
            
            self.dw['dW'+str(l)]= dwm      