from ..forward import Sigmoid,ReLU,Tanh,Linear,Identity 
import numpy as np


class SoftmaxCrossEntropy:
    def  __init__ (self , y_out , y_true):

        self.y_true = y_true-1       
        self.y_out = y_out
        self.grads={}
        
    def loss(self):

        sample_loss=0
        label=self.y_true
        
        dl_dz=(np.zeros(len(self.y_out))).reshape(len(self.y_out) , 1)
        last_layer=len(self.layers_num_arr)
        #ind = self.CONST[label]
        a_prelast= self._params['A'+ str(last_layer-1)]
        l_inp=len(a_prelast)
        z=self._params['Z'+ str(last_layer)]
        dW = np.zeros((len(z), len(a_prelast)+1))
        dA_prev = np.zeros((len( z), len(a_prelast)))
        a_prelast= np.append( a_prelast ,1)
        a_prelast=a_prelast.reshape(1 ,len(a_prelast))
        
        w=self._params['W'+ str(last_layer)] 
        act_fc=self.layer_activations[ last_layer]

        for i in range (len(self.y_out)):
            if i ==label:
                dl_dz[i]=self.y_out[i]-1

            else:
                dl_dz[i]=self.y_out[i]
                
                 
       
        #print(self.CONST)
        
        #print(dl_dz)
        #dl_dz[label] = (dl_dz[label])-1
        
        dA_prev = w
        
        dW =np.dot(dl_dz  ,a_prelast)
       
    
        
           
    
        dl_db=dW[:,-1]
        dl_db=dl_db.reshape(len(self.y_out),1)
        #print(dW)
        self.grads['dW'+str(last_layer)] = dW[:,:-1]
        self.grads['db'+str(last_layer)] =dl_db
        self.grads['dA'+str(last_layer)] = np.dot(dA_prev.T , dl_dz).reshape(l_inp ,1)


        sample_loss=-np.log(self.y_out[label])
        print (sample_loss )
        return sample_loss 
    
   
