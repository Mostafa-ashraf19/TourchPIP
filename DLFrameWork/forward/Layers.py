import numpy as np
from itertools import count

# _ids = count(1)
# _params= dict()#{}
# layer_activations= dict()#{}
# layers_num_arr = list()#[]

# layer_activations.clear()
# _params.clear()
# layers_num_arr.clear()
# class info:
#     _ids = count(1)
#     _params= dict()#{}
#     layer_activations= dict()#{}
#     layers_num_arr = list()#[]


class Layer_Dense:
    _ids = count(1)
    _params= dict()#{}
    layer_activations= dict()#{}
    layers_num_arr = list()#[]
 
    # _ids = count(1)
    # _params={}
    # layer_activations={}
    # layers_num_arr = []

    def __init__(self, n_inputs, n_neurons, weight_type="random"):
        '''
        # Initialize weights and biases randomly
        @params :   n_inputs ---> number of features
                    n_neurons --> number of neurons
        '''
        # print('Hello from constuctor')
        # tracking number of instances
        # _ids = count(1)

        self.layer_number = next(Layer_Dense._ids)
        Layer_Dense.layers_num_arr.append(self.layer_number)
        # self._params= []#{}
        # self.activations = []
        # self.layernum = 0
        # this->x = 1
        
        # Initializing the weights either random or zeros
        if weight_type == "random":
            self.weights = 0.1 * np.random.randn( n_neurons,n_inputs)
        elif weight_type == "zeros":
           self.weights = np.zeros(( n_neurons, n_inputs))

        self.biases = np.zeros((n_neurons, 1))

        # adding the initialized weight to _params dictionary
        Layer_Dense._params['W'+ str(self.layer_number)] = self.weights
        Layer_Dense._params['b'+ str(self.layer_number)] = self.biases
    def __del__(self):
        Layer_Dense._params.clear()
        Layer_Dense.layer_activations.clear()
        Layer_Dense.layers_num_arr.clear() #= 5#.clear()
        Layer_Dense._ids = count(1)


    #   passing data from activation function classes to Layer_Dens class
    def pass_act(self,act_type: str , A):
        """
        description 
        @Param: ---
        @Return: ---
        """
        Layer_Dense._params['A'+ str(self.layer_number)] = A
        Layer_Dense.layer_activations[self.layer_number] = act_type

    # multibly W*X and save them to out and _params dictionary
    def forward(self,inputs):
        if self.layer_number == 1: 
            # print('1st print', inputs.shape) 
            Layer_Dense._params['A0'] = inputs
        self.out = np.dot(self.weights, inputs) + self.biases # Z
        # self._params['Z' + str(self.layer_number)]= self.out

        return (self.out, self.weights,self.biases) 
        
    # get weights
    def get_weights(self):
        return self.weights

    # set weights and save them to _params dictionary
    def set_weights(self , new_weights):
        self.weights = new_weights
        Layer_Dense._params['W'+ str(self.layer_number)] = self.weights
        Layer_Dense._params['b'+ str(self.layer_number)] = self.biases

    def __reper__(self):
        pass



# class Layer_Dense:
 
#     # _ids = count(1)
#     # _params={}
#     # layer_activations={}
#     # layers_num_arr = []

#     def __init__(self, n_inputs, n_neurons, weight_type="random"):
#         '''
#         # Initialize weights and biases randomly
#         @params :   n_inputs ---> number of features
#                     n_neurons --> number of neurons
#         '''

#         # tracking number of instances
#         layer_number = next(self._ids)
#         layers_num_arr.append(self.layer_number)
#         # self._params= []#{}
#         # self.activations = []
#         # self.layernum = 0
#         # this->x = 1
        
#         # Initializing the weights either random or zeros
#         if weight_type == "random":
#             self.weights = 0.1 * np.random.randn( n_neurons,n_inputs)
#         elif weight_type == "zeros":
#            self.weights = np.zeros(( n_neurons, n_inputs))

#         self.biases = np.zeros((n_neurons, 1))

#         # adding the initialized weight to _params dictionary
#         _params['W'+ str(self.layer_number)] = self.weights
#         _params['b'+ str(self.layer_number)] = self.biases

#     #   passing data from activation function classes to Layer_Dens class
#     def pass_act(self,act_type: str , A):
#         """
#         description 
#         @Param: ---
#         @Return: ---
#         """
#         _params['A'+ str(self.layer_number)] = A
#         layer_activations[self.layer_number] = act_type

#     # multibly W*X and save them to out and _params dictionary
#     def forward(self,inputs):
#         if self.layer_number == 1: 
#             print('1st print', inputs.shape) 
#             _params['A0'] = inputs
#         self.out = np.dot(self.weights, inputs) + self.biases # Z
#         # self._params['Z' + str(self.layer_number)]= self.out

#         return (self.out, self.weights,self.biases) 
        
#     # get weights
#     def get_weights(self):
#         return self.weights

#     # set weights and save them to _params dictionary
#     def set_weights(self , new_weights):
#         self.weights = new_weights
#         _params['W'+ str(self.layer_number)] = self.weights
#         _params['b'+ str(self.layer_number)] = self.biases

#     def __reper__(self):
#         pass
