from DLFrameWork.forward import Layer_Dense 
from DLFrameWork.forward import Sigmoid 
from DLFrameWork.losses import SD, Biploar_SVM ,BiploarPerceptron
import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt
# h5py, numpy , pandas, matplotlib


def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

if __name__ == '__main__':
    # z = np.array([1.26428867])
    # print(Sigmoid.sigmoid_(z))  
    # print(Sigmoid.sigmoidBW_(z))
    train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
    index = 10
    plt.imshow(train_x_orig[index])
    print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
    plt.show()
    # X = np.array([1,1,0,0,1,0,0,1,0,0,1,0]) # data  (12*1)
    # X = X.reshape(-1,1)
    # layer1 = Layer_Dense(12,3) # layer1 
    # sig_inst = Sigmoid() # sigmoid
    # layer2 = Layer_Dense(3,1) # layer2 

    # layer1.set_weights (np.array([  # W1 (3*12)
    #     [0.48,0.29,0.58,0.61,0.47,0.18,0.02,0.49,0.67,0.62,0.017,0.28],
    #     [0.52,0.036,0.90,0.099,0.46,0.87,0.99,0.83,0.15,0.14,0.64,0.88],
    #     [0.48,0.055,0.56,0.16,0.86,0.34,0.40,0.06,0.66,0.72,0.077,0.29]
    # ]))
    # layer2.set_weights(np.array([  # W2 
    #     [0.33,0.21,0.96]
    # ]))
  
    # layer1.forward(X) # Z1 = W1.X+b1 
    # # print('Z1 output ',layer1.out)
    # # print('-'*20)
    # A1 = sig_inst.forwards(layer1) # A1 = sigmoid(Z1)
    # # A1 = sig_inst.sigmoid()
    # # print('A1 output',A1)
    # # print('-'*20)

    # layer2.forward(A1) # Z2 = W2.A1 + b2
    # # print('Z2 output',layer2.out)
    # # print('-'*20)

    # A2 = sig_inst.forwards(layer2) # A2 = sigmoid(Z2)
    # # print('A2 output',A2)
    # # print('-'*20)
    
    # # print(Layer_Dense._params) # W1,b1,Z1,A1,W2,b2,Z2,A2   
    # # print('-'*20)
    # # print(Layer_Dense.layer_activations) # activation functions
    
    # Y = np.array([1])
    # loss = SD(A2,Y)
    # # loss = Biploar_SVM(A2,Y)
    # # loss = BiploarPerceptron(A2,Y)
    # # print('loss {}'.format(loss.loss()))
    # # print('-'*20)

    # # loss.backward()
    # # print(loss._params) 

    # new_pa = loss.StepBackward(0.1)
    # Layer_Dense._params= new_pa[0]

    # print('new param is ',new_pa[0])

    # print('----'*10)

    # new_pa1 = loss.StepBackward(0.1)

    # print('new param 2  is', new_pa1[0])
    # loss = 1
    


    # # while loss

   
   


