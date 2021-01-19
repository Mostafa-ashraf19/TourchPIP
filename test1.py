from DLFrameWork.forward import Layer_Dense 
from DLFrameWork.forward import Sigmoid 
from DLFrameWork.losses import SD, Biploar_SVM ,BiploarPerceptron,BiploarPerceptron
import numpy as np

if __name__ == '__main__':
    X = np.array([1,1,0,0,1,0,0,1,0,0,1,0]) # data  (12*1)
    X = X.reshape(-1,1)
    layer1 = Layer_Dense(12,3) # layer1 
    sig_inst = Sigmoid() # sigmoid
    layer2 = Layer_Dense(3,1) # layer2 

    layer1.set_weights (np.array([  # W1 (3*12)
        [0.48,0.29,0.58,0.61,0.47,0.18,0.02,0.49,0.67,0.62,0.017,0.28],
        [0.52,0.036,0.90,0.099,0.46,0.87,0.99,0.83,0.15,0.14,0.64,0.88],
        [0.48,0.055,0.56,0.16,0.86,0.34,0.40,0.06,0.66,0.72,0.077,0.29]
    ]))
    layer2.set_weights(np.array([  # W2 
        [0.33,0.21,0.96]
    ]))
  
    layer1.forward(X) # Z1 = W1.X+b1 
    # print('Z1 output ',layer1.out)
    # print('-'*20)
    A1 = sig_inst.forwards(layer1) # A1 = sigmoid(Z1)
    # print('A1 output',A1)
    # print('-'*20)

    layer2.forward(A1) # Z2 = W2.A1 + b2
    print('Z2 output',layer2.out)
    print('-'*20)

    A2 = sig_inst.forwards(layer2) # A2 = sigmoid(Z2)
    print('A2 output',A2)
    print('-'*20)
    
    # print(Layer_Dense._params) # W1,b1,Z1,A1,W2,b2,Z2,A2   
    # print('-'*20)
    # print(Layer_Dense.layer_activations) # activation functions
    # print('-'*20)
    
    Y = np.array([1])
    # loss = SD(A2,Y)
    # loss = Biploar_SVM(A2,Y)
    loss = BiploarPerceptron(A2,Y)
    print('loss {}'.format(loss.loss()))
    # print(loss._params) 


    # loss.backward()

   


