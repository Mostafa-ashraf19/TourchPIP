import numpy as np
from ..forward import Layer_Dense
from ..forward import Sigmoid,ReLU,Tanh,Identity

class BiploarPerceptron(Layer_Dense):
    def __init__(self,pred,lable):
        self.pred = pred
        self.lable = lable
        self.parameters = Layer_Dense._params #parameter
        self.dl={}
    def loss(self):
        return max(0,-(self.lable+self.parameters['Z'+str(len(Layer_Dense.layer_activations.keys()))]))
    def preceptron_loss_grad(self):
        L = len(Layer_Dense.layer_activations.keys())
        act_fc=Layer_Dense.layer_activations[L]

        if (self.lable*self.parameters['Z'+str(L)] )> 0:
            self.dl['dW'+str(L)] = np.zeros(self.parameters['W'+str(L)].shape)
            self.dl['db'+str(L)] = np.zeros(self.parameters['b'+str(L)].shape)
            self.dl['dA'+str(L)] = np.zeros(self.parameters['A'+str(L)].shape)
                 
        else :
            dl_activation=np.zeros(self.parameters['Z'+str(L)].shape)
            if act_fc == 'sigmoid':
               sig_inst=Sigmoid()
               dl_activation = sig_inst.backwards(self.parameters['Z'+str(L)]) 
               print(dl_activation)
  
            elif act_fc == 'relu':
                relu_inst =ReLU()
                dl_activation= relu_inst.Backwards(self.parameters['Z'+str(L)])
            elif act_fc == 'tanh':
                tanh_inst = Tanh()
                dl_activation= tanh_inst.backwards(self.parameters['Z'+str(L)])
            
            dl_dz = np.matmul(-self.lable,dl_activation)
            self.dl['dW'+str(L)] = np.matmul(self.parameters['W'+str(L)],dl_dz)
            self.dl['db'+str(L)] = dl_dz
            # self.dl['dA'+str(l)] = np.matmul(self.lable,act_fc.backwards(self.parameters['Z'+str(L)]))
            

# class BiploarPerceptron(Layer_Dense):
#     def __init__(self,pred,lable):
#         self.pred = pred
#         self.lable = lable
#         self.parameters = Layer_Dense._params #parameter
#         self.dw={}
#     def loss(self):
#         return max(0,-(self.lable+self.parameters['Z'+str(len(Layer_Dense.layer_activations.keys()))]))
#     def preceptron_loss_grad(self):
#         dwm =[]
#         L =len(self.lable)
#         for l in reversed(range(L)): 
#             if (self.lable[l]*np.dot(self.parameters['w'+str(l)].shap,self.data[l])) > 0:
#               dwm[l]= np.zeros(self.parameters['w'+str(l)].shape)
#             else:
#               for j in range (0,L):
#                   dwm[j] = np.dot(self.lable[l],self.data[j])  
#             self.dw['dw'+str(l)]= dwm    