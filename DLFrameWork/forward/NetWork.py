from .Linear import Linear
from . import Sigmoid
from . import ReLU,Sigmoid,Tanh,Softmax,Identity,Softmax
from itertools import repeat
from ..losses import SD#, Biploar_SVM ,BiploarPerceptron
from ..backward import Backward
import numpy as np
# from 
from ..losses import SoftmaxCrossEntropy
from ..optimizations import Optimizer

### combination ---->>> 1- network one time -- > train ,  pred, plot 
#2- network on steps 



def lossgrad(A2,Y):
    return - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

def compute_cost(AL, Y):
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

def costsum(cost):
    # print('Hello cost sum')
    return np.sum(cost)


def _tupleDivide(tup):
    layersShape =  list()    
    for i in range(len(tup)-1):
        _tup = (tup[i],tup[i+1])
        layersShape.append(_tup)
    return layersShape

class NetWork:
    def __init__(self,LayersShape,activations,optimType={'GD':True}):
        self.LayersShape = LayersShape
        print(_tupleDivide(LayersShape))
        self.createdLayers = [Linear(layer[0],layer[1],l_size=layer[0]) for layer in _tupleDivide(LayersShape)]
        self.activations = list(repeat(activations, len(self.createdLayers))) if type(activations) == str \
                                                else activations 

        self.LayersLen = len(self.createdLayers) 
        self.parameters = dict()
        self.Zout = list()
        self.Aout = list()
        self.lossvalues = []
        self.optimType  = optimType 
        # self.McreatedLayers

    def fit(self,X,Y,learning_rate=0.1):#,lossClass):
        ########## Forward L Step        
        self._CalcFWD(X)
        self._ConstructParams() 
        # print('parameters is',self.parameters)

        ############## Lossess
        # cost = SD._Loss(self.Aout[self.LayersLen-1],Y)
        # cost = SD._Cost(self.Aout[self.LayersLen-1],Y)
        # print(self.Aout[self.LayersLen-1],Y)
        # print('shapes is ',self.Aout[self.LayersLen-1].shape,'', Y.T.shape)
        cost = SoftmaxCrossEntropy._Cost(self.Aout[self.LayersLen-1],Y)
        
        # print('Cost is',cost)
        # cost = compute_cost(self.Aout[self.LayersLen-1],Y)
        # loss = lossClass._Loss(self.Aout[self.LayersLen-1],Y)


        # lossD = SD._Grad(self.Aout[self.LayersLen-1],Y)
        # print('loss D', lossD)
        # lossD = lossgrad(self.Aout[self.LayersLen-1],Y)
        lossD = SoftmaxCrossEntropy._Grad(self.Aout[self.LayersLen-1],Y.T)
        # lossD = 10
        # Y label 
        # print('loss drvative is ', lossD.shape)
        # lossD = lossClass._Grad(self.Aout[self.LayersLen-1],Y)
        
        
        ########## backward 
        grads = Backward.StaticBW(self.parameters,learning_rate=learning_rate,LossDerivative=lossD,
                    Y=Y,layersLen=self.LayersLen,layer_activations=self.activations)
        ###########
        
        ######### parameters update
        optimizer = Optimizer(parameters=self.parameters,grads=grads,optimiType=self.optimType,LayersLen=self.LayersLen,lr=learning_rate)
        # self.parameters = self._optimizerUsage(self.optimizations,grads=grads,lr=learning_rate)
        self.parameters = optimizer.optimize()
        # print(self.parameters)
        self._UpdateLayerParam()
        ###########################

        # print('parameters is ',self.parameters)            
        self.Aout.clear()
        self.Zout.clear()
        # self.lossvalues.append(cost)
        return cost  
    

    def Parameters(self):
        # self.ActivationCalc
        return self.parameters    
    def _updateA_Z(self):
        for i in range(len(self.createdLayers)):
            self.Zout[i] = self.parameters['Z'+str(i+1)]
            self.Aout[i] = self.parameters['A'+str(i+1)]


    def _UpdateLayerParam(self):
            # print('update layer params')
        for i in range(self.LayersLen):
            self.createdLayers[i].updateW_B(self.parameters['W'+str(i+1)],self.parameters['b'+str(i+1)])
            # print('weights and bias is {}'.format(self.createdLayers[i].__reper__()))

    def _ConstructParams(self):
        for i in range(self.LayersLen):
            self.parameters['W'+str(i+1)],self.parameters['b'+str(i+1)] = self.createdLayers[i].__reper__()
            self.parameters['Z'+str(i+1)] = self.Zout[i]
            self.parameters['A'+str(i+1)] = self.Aout[i]
        
            
        # pass

    def _CalcFWD(self,X):
        # pass
        aOut = X
        self.parameters['A0'] = X
        # print('X is ', X)
        for Layer,activation in zip(self.createdLayers,self.activations):
            # print('layer is {}, activation is {}'.format(Layer.__reper__(),activation))
            zOut = Layer.forward(aOut)
            self.Zout.append(zOut)
            # print('zOut is', zOut)
            aOut = NetWork.ActivationCalc(zOut,activation)
            # print('aout is', aOut)

            self.Aout.append(aOut) 
    
    def FWD_predict(self,X):
        aOut = X
        for Layer,activation in zip(self.createdLayers,self.activations):
            zOut = Layer.forward(aOut)
            aOut = NetWork.ActivationCalc(zOut,activation)
        return aOut  

    def Prediction(self,X,Y,parameter):
        if  self.LayersShape[-1] > 1:
            return self._MultiPrediction(X,Y,parameter)
        return self._BinaryPrediction(X,Y,parameter)
    # multi classfication       
    def _MultiPrediction(self,X,Y,parameter):
        m = X.shape[1]
        p = list()#np.zeros((1,m))
        print('m is ', m )
        probas = self.FWD_predict(X)
        print('prob is {}, label is {}'.format(probas,Y))
        for i in range(0, probas.shape[1]):
            m = max(probas[:,i])
            max_indexes=[s for s, t in enumerate(probas[:,i]) if t == m]
            # max_indexes = np.array(max_indexes)
            p.append(max_indexes[0])
    
        print("Accuracy: "  + str(np.sum((p == Y)/m)))
        return p
    # binay classfication    
    def _BinaryPrediction(self,X,Y,parameter):
        m = X.shape[1]
        n = self.LayersLen # number of layers in the neural network
        p = np.zeros((1,m))
        
        # Forward propagation
        probas = self.FWD_predict(X)

        # print('Y is ', Y)
        # convert probas to 0/1 predictions
        # print('probs from pred is', probas)
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                # print("Ture")
                p[0,i] = 1
            else:
                # print("False")
                p[0,i] = 0
        
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        print("Accuracy: "  + str(np.sum((p == Y)/m)))
        return p
        # pass 
    @staticmethod
    def ActivationCalc(zOut,activation='ReLU'):
        if activation == 'ReLU':
            # print('Hello from RELU')
            return ReLU.ReLU_(zOut)
        elif activation == 'Sigmoid':
            # print('Hello from Sigmoid')
            return Sigmoid.sigmoid_(zOut)
        elif activation == 'Tanh':
            return Tanh.Tanh_(zOut) 
        elif activation == 'SoftMax':
            return Softmax.Softmax_(zOut)    
        # elif activation       
        # AssertionError ()
        assert 'No Activation Setted\n'         

    def PlotLoss(self):
        pass   



    def Linear(self,n_inputs, n_neurons, weight_type="random"):
        self.McreatedLayers.append(Linear(n_inputs, n_neurons, weight_type="random"))
        return self.McreatedLayers[0]
        pass
            
    def save_model(self,path):
        pass
