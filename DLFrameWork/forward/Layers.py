import numpy as np
from itertools import count

class Layer_Dense:
    
    _ids = count(1)
    _params={}
    layer_activations={}
    layers_num_arr = []
    


    def __init__(self, n_inputs, n_neurons, weight_type="random"):
        '''
        # Initialize weights and biases randomly
        @params :   n_inputs ---> number of features
                    n_neurons --> number of neurons
        '''

        # tracking number of instances
        self.layer_number = next(self._ids)
        self.layers_num_arr.append(self.layer_number)
        
        # Initializing the weights either random or zeros
        if weight_type == "random":
            self.weights = 0.1 * np.random.randn( n_neurons,n_inputs)
        elif weight_type == "zeros":
           self.weights = np.zeros(( n_neurons, n_inputs))

        self.biases = np.zeros((n_neurons, 1))

        # adding the initialized weight to _params dictionary
        self._params['W'+ str(self.layer_number)] = self.weights
        self._params['b'+ str(self.layer_number)] = self.biases

    #   passing data from activation function classes to Layer_Dens class
    def pass_act(self,act_type: str , Z):
        self._params['Z'+ str(self.layer_number)] = Z
        self.layer_activations[self.layer_number] = act_type

    # multibly W*X and save them to out and _params dictionary
    def forward(self,inputs):
        self.out = np.dot(self.weights, inputs) + self.biases
        self._params['A' + str(self.layer_number)]= self.out
        
    # get weights
    def get_weights(self):
        return self.weights

    # set weights and save them to _params dictionary
    def set_weights(self , new_weights):
        self.weights = new_weights
        self._params['W'+ str(self.layer_number)] = self.weights
        self._params['b'+ str(self.layer_number)] = self.biases