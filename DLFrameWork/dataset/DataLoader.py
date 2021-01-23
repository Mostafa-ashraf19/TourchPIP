import pandas
import numpy as np

class DataLoader:

    def __init__(self,path='',batchsize=20,shuffling=True, normalization={}):
        """
        initializing the params
        @params: path
                batchsize      ---> number of raws loaded from csv
                shuffling      ---> shuffle raws indexs
                nomaliztion    ---> normalize between 0-1

        @return: 1) features raws
                2) labels raws
        """
        self.path = path
        self.current_position = 0
        self.batchsize = batchsize
        self.shuffling = shuffling
        self.normalization = normalization['Transform'] if type(normalization['Transform']) == bool else RuntimeError('should be bool value ') 
        # self.transform = transform


    def __itr__(self):
        return self

    def __next__(self):
           
            df = pandas.read_csv(self.path, skiprows=self.current_position,nrows=self.batchsize)  
            self.current_position += self.batchsize

            if self.normalization == False:             #if No normalization

                if not self.shuffling:                  #if No shuffling
                    return df.iloc[:, 1:].to_numpy(), df.iloc[:, 0].to_numpy()  

                else:                                   #if shuffling
                    x = df.iloc[:, 1:].to_numpy()
                    y = df.iloc[:, 0].to_numpy()
                    np.random.shuffle(x)
                    np.random.shuffle(y)
                    return x , y
                
            else:                                       #if normalization
                if not self.shuffling:                  #if No shuffling
                    x = df.iloc[:, 1:].to_numpy()
                    norm_x = (x - np.min(x))/np.ptp(x)
                    return norm_x , df.iloc[:, 0].to_numpy()

                else:                                   #if shuffling
                    x = df.iloc[:, 1:].to_numpy()
                    norm_x = (x - np.min(x))/np.ptp(x)
                    y = df.iloc[:, 0].to_numpy()
                    np.random.shuffle(norm_x)
                    np.random.shuffle(y)
                    return norm_x , y



