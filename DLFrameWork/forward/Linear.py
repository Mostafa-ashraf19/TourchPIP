import numpy as np
# from itertools import count


class Linear:
    
    def __init__(self, n_inputs, n_neurons,l_size, weight_type="random"):
        '''
        # Initialize weights and biases randomly
        @params :   n_inputs ---> number of features
                    n_neurons --> number of neurons
        '''
        # print('Hello from constuctor')     
        # Initializing the weights either random or zeros
        if weight_type == "random":
            # print('Hello from random weights')     
            # np.random.seed(1)
            self.weights = 0.1 * np.random.randn(n_neurons,n_inputs)
        elif weight_type == "random_l":
            self.weights = np.random.randn(n_neurons,n_inputs) * np.sqrt(2/l_size)     
        elif weight_type == "zeros":
           self.weights = np.zeros(( n_neurons, n_inputs))

        self.biases = np.zeros((n_neurons, 1))

        
    # def __del__(self):
    #     self.weights.clear()
    #     self.biases.clear()


   
    def forward(self,X):
        # print('shape of w is {}, shape of x if {}'.format(self.weights.shape,X.shape))
        Z = np.dot(self.weights,X) + self.biases # Z
        return Z 
    # # get weights
    # def get_weights(self):
    #     return self.weights

    # set weights and save them to _params dictionary
    def set_weights(self , new_weights):
        self.weights = new_weights
        # self._params['W'+ str(self.layer_number)] = self.weights
        # self._params['b'+ str(self.layer_number)] = self.biases
    def updateW_B(self,weights,bias):
        # print('Hello from linear update w and b')
        self.weights = weights 
        self.biases = bias
    def __reper__(self):
        return self.weights, self.biases
    def Values(self):
        return self.weights, self.biases
