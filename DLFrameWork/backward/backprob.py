import numpy as np 
from ..forward import Layer_Dense
from ..forward import ReLU,Sigmoid,Tanh,Softmax,Identity

class Backward:  # make inheratnce to loss, and layers 
    def __init__(self,Y):
        # self.AL = AL
        self.Y = Y
        self.grads = {} #grads 
        self.parameters = Layer_Dense._params #parameters 
        self.learning_rate = 0.1
        self.LossDerivative = 0
        #print('from backprop',Layer_Dense._params)
        self.caches = { i: (Layer_Dense._params['A'+str(i)],Layer_Dense._params['W'+str(i)],Layer_Dense._params['b'+str(i)]) if i != 0 else 
                                (Layer_Dense._params['A'+str(i)],'_') for i in range(0,int(((len(Layer_Dense._params.keys())-1)/3)+1))}
 
    # def _L_model_backward(self,AL, Y, caches):
    def _L_model_backward(self):# caches):
        
        L = len(Layer_Dense.layers_num_arr) # the number of layers
        #print('num of layyers is ',L)
        m = self.caches[L][0].shape[1] #self.AL.shape[1]
        self.Y = self.Y.reshape(self.caches[L][0].shape)#(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = self.LossDerivative # loss dervatie (dl/dy)
        self.grads['dA'+str(L)] = dAL
        # print('Shape of DAL',dAL.shape)
        current_cache = (self.caches[L-1][0],self.caches[L][1],self.caches[L][2]) 
        # print('DZ2', self.grads['dZ'+str(L)].shape)
        # print('-'*10,'\n done from fun', '-'*10)
        self.grads["dA" + str(L-1)], self.grads["dW" + str(L)], self.grads["db" + str(L)] = self._LinearActivaionBW(dAL,
                                    L,linear_cache=current_cache,activation=Layer_Dense.layer_activations[L])
        # print('grads eq','-'*10,'\n',self.grads)
        # exit()    
        # Loop from l=L-2 to l=0
        # print('L before loop', L)
        for l in reversed(range(1,L)):
            # print('cache from loop', self.caches[0])
            # print('l from loop', l)
            #print('from loop ', self.caches[l-1][0].shape)
           # print('-*'*10,(self.caches[l-1][0].shape,self.caches[l][1],self.caches[l][2]))
            current_cache = (self.caches[l-1][0],self.caches[l][1],self.caches[l][2])#self.caches[l]

            dA_prev_temp, dW_temp, db_temp =  self._LinearActivaionBW(self.grads['dA'+str(l)],
                                    l,linear_cache=current_cache,activation=Layer_Dense.layer_activations[l])
            self.grads["dA" + str(l-1)] = dA_prev_temp
            self.grads["dW" + str(l)] = dW_temp
            self.grads["db" + str(l)] = db_temp

        # return self.grads  

    def backward(self):
        self._L_model_backward()
        return self._update_parameters(), self.grads
    

    def _LinearActivaionBW(self,dAL,l,linear_cache,activation='ReLU'):
        dZ = self._bacwardActivations(dAL,l=l,activation=activation) # (1,1)dz2
        dA_prev, dW, db = self._linear_backward(dZ,linear_cache)# da1,dw2,db2
        return dA_prev,dW,db

    def _bacwardActivations(self,dAL,l,activation='Relu'):        
        if activation == 'Relu':
            self.grads['dZ'+str(l)] = ReLU.ReLUBW_(dAL)  
        elif activation == 'Sigmoid':
            # print('Hello from sigmoid')
            self.grads['dZ'+str(l)] = Sigmoid.sigmoidBW_(dAL)
            return self.grads['dZ'+str(l)]
        elif activation == 'Tanh':
            self.grads['dZ'+str(l)] = Tanh.TanhBW_(dAL)

    def _linear_backward(self,dZ, cache):
        
        # print('cache in linear is', cache)

        A_prev, W, b = cache
        #print('A_prev shape',A_prev.shape)
        m = A_prev.shape[1]
        # print('DZ is', dZ)
        # print('DZ dim', dZ.shape, 'A_prev dim', A_prev.shape)
        dW = (1/m) * np.matmul(dZ,A_prev.T) # dz2 (1,1) * a1 (3*1)
        db = (1/m) * np.sum(dZ,axis=1,keepdims=True)
        # print('W shape ',W.shape,'Dz.shape',dZ.shape )

        dA_prev = np.matmul(W.T,dZ) # da1 = w2.t(3,1) . dz2(1,1) 
        # print('dA_prev shape ',dA_prev.shape,'A_prev.shape', A_prev.shape )
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db

    def _update_parameters(self):
               
        L = len(Layer_Dense.layers_num_arr) // 2 # number of layers in the neural network
        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            self.parameters["W" + str(l+1)] =  self.parameters["W" + str(l+1)] - self.learning_rate * self.grads['dW'+str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - self.learning_rate * self.grads['db'+str(l+1)]
        return self.parameters  