import numpy as np
from ..forward import Layer_Dense


class SD(Layer_Dense):
    def __init__(self,pred,lable):
        self.pred = pred
        self.lable = lable
        # self.parameters = parameter#Layer_Dense._params #parameter
        self.dw={}

    def loss(self) :    
      pred_minus_lable = np.subtract(self.pred , self.lable)
      pred_minus_lable_T = pred_minus_lable.T
      return 0.5 * np.dot(pred_minus_lable_T, pred_minus_lable)

    def SQD_loss_grade(self) :
          return np.subtract(self.pred , self.lable)