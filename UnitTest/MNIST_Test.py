from DLFrameWork.forward import NetWork 
from DLFrameWork.dataset import FashionMNIST,DataLoader

if __name__ == '__main__':
    FMNIST = FashionMNIST(path='MNIST_Data',download=False,train=True)
    
    dLoader = DataLoader(FMNIST,batchsize=100,shuffling=False,normalization={'Transform':True})
    # (784,256),(256,128),(128,64),(64,10)
    net = NetWork((784,256,128,64,10),('ReLU','ReLU','ReLU','SoftMax'),optimType={'Momeuntum':True})
    
    print(net)
    costs = []
    print_cost = True
    epochs = 10
    for i in range(epochs):  
        cost = 0.0
        for j,(images,labels) in enumerate(dLoader):
            ourimages = images.T 
            ourlabel = labels.T
            innercost = net.fit(ourimages,ourlabel,learning_rate =0.02)
            cost += innercost
            # print('iteration num {},inner cost is {}'.format(j, innercost))
        if print_cost:# and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, cost/600))
            print('-'*10)

    images, labels = next(dLoader)
    
    net.Prediction(images.T,labels.T,net.Parameters())	
    
