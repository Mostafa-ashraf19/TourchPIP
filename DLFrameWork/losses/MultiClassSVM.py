from ..forward import Sigmoid,ReLU,Tanh,Linear,Identity 
import numpy as np


class MultiClassSVM: #(Layer_Dense): #y_true, y_out , a_prelast , w, act_fc , b
    def  __init__ (self , y_out , y_true):
        self.y_true=y_true-1
        self.y_out=y_out
        self.grads={}   

    @staticmethod
    def loss(AL,ZL,y_true,layers_num_arr,_params,layer_activations):
        sample_loss=0
        count = 0
        last_layer=len(layers_num_arr)
        a_prelast= _params['A'+ str(last_layer-1)] 
        z=_params['Z'+ str(last_layer)]
        dW = np.zeros((len(z), len(a_prelast)+1))
        dA_prev = np.zeros((len( z), len(a_prelast)))
        a_prelast= np.append( a_prelast ,1).T
        w= _params['W'+ str(last_layer)] 
        act_fc= layer_activations[ last_layer]
        dl_dz=(np.zeros(len(z))).reshape(len(z) , 1)

        label = y_true

        if act_fc == Sign or act_fc == Identity or act_fc == None :

        
            for i in range(len(z)):

                if max(1+z[i]-z[label] , 0) and i!= label:
                    count+=1
                    dW[i]=a_prelast
                    dA_prev[i] =w[i]
                    dl_dz[i] =1
                    sample_loss+=1+z[i]-z[label]

                
    
            if count >0:

                dW[label]= -count * a_prelast
                dA_prev[label]=  w[label]
                dl_dz[label] = -count
                


            dl_db=dW[:,-1]
            dl_db=dl_db.reshape(len(z),1)
            grads['dW'+str(last_layer)] = dW[:,:-1]
            grads['db'+str(last_layer)] =dl_db
            grads['dA'+str(last_layer)] = np.dot(dA_prev.T , dl_dz).reshape(len(a_prelast)-1 ,1)


            

        else:
            for i in range(len(y_out)):

                if max(1+y_out[i]- y_out[label] , 0) and i!= label:
                    count+=1
                
                    dl_dz[i] =act_fc.backwards(y_out[i])

                    dW[i] =  a_prelast * dl_dz[i]
                    dA_prev[i] = w[i] 
                    sample_loss+=1+ y_out[i] - y_out[label]    
            if count >0:

                dl_dz[label] =act_fc.backwards(y_out[label]) * -count

                dW[label] =  a_prelast * dl_dz[label]
                dA_prev[label] = w[label] 
            dl_db=dW[:,-1]
            dl_db=dl_db.reshape(len(z),1)
            grads['dW'+str(last_layer)] = dW[:,:-1]
            grads['db'+str(last_layer)] =dl_db
            grads['dA'+str(last_layer)] = (np.dot(dA_prev.T , dl_dz)).reshape(len(a_prelast)-1 ,1)        
        return sample_loss 
    

