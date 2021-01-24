from DLFrameWork.forward import NetWork 
from DLFrameWork.dataset import FashionMNIST,DataLoader

if __name__ == '__main__':
    FMNIST = FashionMNIST(path='MNIST_Data',download=True,train=True)
    
    dLoader = DataLoader(FMNIST,batchsize=500,shuffling=True,normalization={'Transform':True})

    net = NetWork((784,256,128,64,10),('ReLU','ReLU','ReLU','SoftMax'))

    costs = []
    print_cost = True
    epochs = 10
    for i in range(epochs): # 1,2 
        cost = 0.0
        for j,(images,labels) in enumerate(dLoader): # 100
            ourimages = images.T #reshape(3072,500)
            ourlabel = labels.T
            innercost = net.fit(ourimages,labels,learning_rate = 0.1)
            cost += innercost
            # print('iteration num {},inner cost is {}'.format(j, innercost))
        if print_cost:# and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost/1000))
            print('-'*10)


    # print(dLoader)
