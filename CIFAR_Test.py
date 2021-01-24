from DLFrameWork.forward import NetWork 
from DLFrameWork.dataset import CIFAR_10,DataLoader

import numpy as np
import h5py
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':

    CiFAR  = CIFAR_10(path='CIFAR_Data',download=False,train=True)
    dLoader = DataLoader(CiFAR,batchsize=500,shuffling=True,normalization={'Transform':True})
    images, labels = next(dLoader)
    # print('shape is ',images[0].shape[0])
    # exit()
    net = NetWork((images[0].shape[0],256,128,64,10),('ReLU','ReLU','ReLU','SoftMax'))
    
    # images,labels = next(dLoader)
    # ourimages = images.T
    # cost = net.fit(ourimages,labels,learning_rate = 0.01)
    # print('cost ', cost)
    # exit()
    costs = []
    print_cost = True
    epochs = 5
    for i in range(epochs): # 1,2 
        cost = 0.0
        for j,(images,labels) in enumerate(dLoader): # 1000
            ourimages = images.T#reshape(3072,500)
            innercost = net.fit(ourimages,labels,learning_rate = 0.01)
            cost += innercost
            print('iteration num {},inner cost is {}'.format(j, innercost))
        if print_cost:# and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost/1000))
            print('-'*10)
        # if print_cost:# and i % 100 == 0:
            # costs.append(cost)
        # print('Iteration {}'.format(i+1))

        # print('shapes is {}, {}'.format(images.shape,labels.shape))
        # print(images[0],' ', labels[0])
        # print('---'*10)
        # i+=1
    # print('i is ', i)    
    