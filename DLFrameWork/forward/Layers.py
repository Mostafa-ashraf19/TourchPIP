import numpy as np
from itertools import count

class Layer_Dense:

    _ids = count(1) #count(layer_number +1 )
    _inputs ={}
    _params={}
    layer_activations={}
    num_of_layer = 0


    def __init__(self, n_inputs, n_neurons, weight_type="random" , ):
        '''
        # Initialize weights and biases randomly
        @params :   n_inputs ---> number of features
                    n_neurons --> number of neurons
        '''
        self.layer_number = next(self._ids)
        self.num_of_layer = self.layer_number
        #------------------------------------
        if weight_type == "random":
            self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        elif weight_type == "zeros":
           self.weights = np.zeros((n_inputs, n_neurons))


       
        self.biases = np.zeros((1, n_neurons))
        self._params['W'+ str(self.layer_number)] = self.weights
        self._params['B'+ str(self.layer_number)] = self.biases

    def pass_act(self,act_type: str):
        self.layer_activations[self.layer_number] = act_type

    def forward(self,inputs : list) -> list:
        self.out = np.dot(inputs, self.weights) + self.biases
        self._params['I'+ str(self.layer_number)] = inputs
        self._params['O' + str(self.layer_number)]= self.out
    
    def get_weights(self) -> list:
        return self.weights

    def set_weights(self , new_weights : list ):
        self.weights = new_weights
        self._params['W'+ str(self.layer_number)] = self.weights
        self._params['B'+ str(self.layer_number)] = self.biases
