import numpy as np 
from ..forward import Layer_Dense#,_params,layers_num_arr,layer_activations
from ..forward import ReLU,Sigmoid,Tanh,Softmax,Identity
from ..losses import SoftmaxCrossEntropy
from ..optimizations import Momentum,Adam


class Backward:  # make inheratnce to loss, and layers 
    def __init__(self,Y,learning_rate,LossDerivative):
        # self.AL = AL
        self.Y = Y
        self.grads = {} #grads 
        self.parameters = dict(Layer_Dense._params) #Layer_Dense._params #parameters 
        self.layers_num_arr  = list(Layer_Dense.layers_num_arr)
        self.layer_activations = dict(Layer_Dense.layer_activations)
        self.learning_rate = learning_rate
        self.LossDerivative = LossDerivative
        #print('from backprop',Layer_Dense._params)
        # print('self.parameters from jupyter',self.parameters)
        self.caches = { i: (self.parameters['A'+str(i)],self.parameters['W'+str(i)],self.parameters['b'+str(i)]) if i != 0 else 
                                (self.parameters['A'+str(i)],'_') for i in range(0,int(((len(self.parameters.keys())-1)/3)+1))}
        # self._params = 5
        # print('self._params ', self._params)
    # def _L_model_backward(self,AL, Y, caches):
    def _L_model_backward(self):# caches):
        # print('_params inside L_model', self.parameters)
        
        # print('parameters inside L_model', self.parameters)
        # print('len(self.layers_num_arr)', len(self.layers_num_arr))
        # print('cache is ', self.caches)
        L = len(self.layers_num_arr) # the number of layers
        #print('num of layyers is ',L)
        m = self.caches[L][0].shape[1] #self.AL.shape[1]
        # print('l size is ', L)
        # print('shape is l ',self.caches[L])
        self.Y = self.Y.reshape(self.caches[L][0].shape)#(AL.shape) # after this line, Y is the same shape as AL
        
        # Initializing the backpropagation
        dAL = self.LossDerivative # loss dervatie (dl/dy)
        self.grads['dA'+str(L)] = dAL
        # print('Shape of DAL',dAL.shape)
        current_cache = (self.caches[L-1][0],self.caches[L][1],self.caches[L][2]) 
        # print('DZ2', self.grads['dZ'+str(L)].shape)
        # print('-'*10,'\n done from fun', '-'*10)
        self.grads["dA" + str(L-1)], self.grads["dW" + str(L)], self.grads["db" + str(L)] = self._LinearActivaionBW(dAL,
                                    L,linear_cache=current_cache,activation=self.layer_activations[L])
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
                                    l,linear_cache=current_cache,activation=self.layer_activations[l])
            self.grads["dA" + str(l-1)] = dA_prev_temp
            self.grads["dW" + str(l)] = dW_temp
            self.grads["db" + str(l)] = db_temp

        # return self.grads  

    def backward(self): # user call it 
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
               
        L = len(self.layers_num_arr) // 2 # number of layers in the neural network
        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            self.parameters["W" + str(l+1)] =  self.parameters["W" + str(l+1)] - self.learning_rate * self.grads['dW'+str(l+1)]
            self.parameters["b" + str(l+1)] = self.parameters["b" + str(l+1)] - self.learning_rate * self.grads['db'+str(l+1)]
        return self.parameters  


    ################################################################
    ####################     Static Methods     ####################
    ################################################################
    @staticmethod
    def _L_Backward(layers_num_arr,parameters,Y,LossDerivative,layer_activations):# caches):   

        caches = { i: (parameters['A'+str(i)],parameters['W'+str(i)],parameters['b'+str(i)]) if i != 0 else 
                                (parameters['A'+str(i)],'_') for i in range(0,layers_num_arr+1)}     
        L = layers_num_arr # the number of layers 2 
        m = caches[L][0].shape[1] #self.AL.shape[1] # 6000 tmaaaaaam 
        # print('AL shape {}, Y shape{}'.format(caches[L][0].shape,Y.shape))
        # Y = Y.reshape(caches[L][0].shape)# msh tmam 
        # after this line, Y is the same shape as AL
        grads = dict()
        # Initializing the backpropagation
        if layer_activations[L-1]  != 'SoftMax':
            dAL = LossDerivative # loss dervatie (dl/dy)
            grads['dA'+str(L)] = dAL # dA2 Henaaaaaa dl/dy === dl/da2
            current_cache = (caches[L-1][0],caches[L][1],caches[L][2]) 
            grads["dA" + str(L-1)], grads["dW" + str(L)],grads["db" + str(L)] = Backward._StaticLinearActivaionBW(dAL,
                                        L,linear_cache=current_cache,activation=layer_activations[L-1],grads=grads)
        else:   
            dZ = SoftmaxCrossEntropy._Grad(caches[L][0],Y)     
            current_cache = (caches[L-1][0],caches[L][1],caches[L][2])
            grads["dA" + str(L-1)], grads["dW" + str(L)],grads["db" + str(L)] = Backward._StaticLinearBW(dZ,current_cache)

        # exit()    
        # Loop from l=L-2 to l=0
        # print('L cap is ', L)
        # print('activations is', list(layer_activations))
        # print('cache is', caches)
        ep = 1
        for l in reversed(range(L-1)):
            # print('l small is ', l)
            current_cache = (caches[l][0],caches[l+1][1],caches[l+1][2])

            dA_prev_temp, dW_temp, db_temp = Backward._StaticLinearActivaionBW(grads['dA'+str(l+1)],
                                    l+1,linear_cache=current_cache,activation=layer_activations[l],grads=grads)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l+1)] = dW_temp
            grads["db" + str(l+1)] = db_temp

        return grads  
    @staticmethod
    def StaticBW(parameters,learning_rate,LossDerivative,Y,layersLen,layer_activations): # user call it 
         # normal gd
        grads = Backward._L_Backward(layers_num_arr=layersLen,parameters=parameters,Y=Y,LossDerivative=LossDerivative,layer_activations=layer_activations)
        # return Backward.StaticParamUpdate(layers_num_arr=layersLen,parameters=parameters,grads=grads,learning_rate=learning_rate)
        # momentum 

        # adam 
        v,s = Adam.initialize_adam(parameters=parameters,layerlen=layersLen)
        return Adam.update_parameters(layerlen=layersLen,v=v,s=s,grads=grads,parameters=parameters,learning_rate=learning_rate)
    @staticmethod
    def _StaticLinearActivaionBW(dAL,l,linear_cache,activation='',grads=''):
        dZ = Backward._StaticBWActivations(dAL,l=l,activation=activation,grads=grads) # (1,1)dz2
        dA_prev, dW, db = Backward._StaticLinearBW(dZ,linear_cache)# da1,dw2,db2
        return dA_prev,dW,db
    @staticmethod
    def _StaticBWActivations(dAL,l,activation='ReLU',grads=''):  
        # print('Hello my activation is', activation)      
        if activation == 'ReLU':
            # print('dAL out', dAL)
            grads['dZ'+str(l)] = ReLU.ReLUBW_(dAL) 
            # print('dZL grads out dervative ',grads['dZ'+str(l)])
            return grads['dZ'+str(l)]
        elif activation == 'Sigmoid':
            grads['dZ'+str(l)] = Sigmoid.sigmoidBW_(dAL)
            return grads['dZ'+str(l)]
        elif activation == 'Tanh':
            grads['dZ'+str(l)] = Tanh.TanhBW_(dAL)
            return grads['dZ'+str(l)]
        elif activation == 'SoftMax':
            grads['dZ'+str(l)] = Softmax.SoftmaxBW_(dAL)
            return grads['dZ'+str(l)]

    @staticmethod
    def _StaticLinearBW(dZ,cache):
        
        # print('cache in linear is', cache)

        A_prev, W, b = cache
        # print('A_prev shape',A_prev.shape)
        m = A_prev.shape[1]

        # print('DZ is', dZ)
        # print('DZ dim', dZ.shape, 'A_prev dim', A_prev.shape)
        # dW = (1/m) * np.matmul(dZ,A_prev.T) # dz2 (1,1) * a1 (3*1)
        # print('DZ shape ',dZ.shape)
        # temp = A_prev.T if A_prev.shape[1] != 
        dW = (1/m) * np.matmul(dZ,A_prev.T) # dz2 (1,1) * a1 (3*1)
        db = (1/m) * np.sum(dZ,axis=1,keepdims=True)
        # print('W shape ',W.shape,'Dz.shape',dZ.shape )

        dA_prev = np.matmul(W.T,dZ) # da1 = w2.t(3,1) . dz2(1,1) 
        # print('dA_prev shape ',dA_prev.shape,'A_prev.shape', A_prev.shape )
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        return dA_prev, dW, db
    @staticmethod
    def StaticParamUpdate(layers_num_arr,parameters,grads,learning_rate):
               
        L = layers_num_arr // 2 # number of layers in the neural network
        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads['dW'+str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads['db'+str(l+1)]
        grads.clear()
        return parameters     