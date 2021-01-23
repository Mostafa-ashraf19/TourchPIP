import numpy as np
from ..forward import Layer_Dense,_params
from ..backward import Backward


class SD(Layer_Dense):
    def __init__(self,pred,lable):
        self.pred = pred
        self.lable = lable
        self.parameters = _params #Layer_Dense._params 
        self.dl={}
        self.backward_ = Backward(Y=self.pred) # instance 

    def loss(self) :    
      pred_minus_lable = np.subtract(self.pred , self.lable)
      pred_minus_lable_T = pred_minus_lable.T
      return 0.5 * np.dot(pred_minus_lable_T, pred_minus_lable)

    def grad(self) :
      return np.subtract(self.pred , self.lable)

    def StepBackward(self,learning_rate):
      self.backward_.learning_rate = learning_rate 
      self.backward_.LossDerivative = self.grad() #  
      return self.backward_.backward()



# class SD(Layer_Dense):
#     def __init__(self,pred,lable):
#         # super(SD,self).__init__(AL=pred,Y=lable,parameters=Layer_Dense._params)
#         self.pred = pred
#         self.lable = lable
#         # self.parameters = parameter#Layer_Dense._params #parameter
#         self.dw={}

#     def loss(self) :    
#       pred_minus_lable = np.subtract(self.pred , self.lable)
#       pred_minus_lable_T = pred_minus_lable.T
#       return 0.5 * np.dot(pred_minus_lable_T, pred_minus_lable)

#     def grade(self) :
#           return np.subtract(self.pred , self.lable)