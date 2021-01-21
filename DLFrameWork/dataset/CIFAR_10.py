import os
import shutil
import tarfile
import urllib.request
import pandas as pd

CIFAR10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


class CIFAR_10:
    def __init__(self, path, download=True, train=True):
        self.path = path
        self.download = download
        self.train = train
        if self.download:
            self._Download()
        self.path = os.getcwd() + '/' + self.path + '/test.csv' if not self.train else os.getcwd() + '/' + self.path + '/train.csv'

    def _Download(self):
        if not os.path.exists(os.getcwd() + '/' + self.path):
            os.mkdir(self.path)
        file_name = 'CIFAR-10.tar.gz'
        with urllib.request.urlopen(CIFAR10_URL) as response, open(os.getcwd() + '/' + self.path + '/' + file_name,
                                                                   'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            tar = tarfile.open(os.getcwd() + '/' + self.path + '/' + file_name, "r:gz")
            tar.extractall(os.getcwd() + '/' + self.path + '/')
            tar.close()

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def toCSV(self):
        file_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        for name in file_names:
            df_labels = pd.DataFrame(self.unpickle('blah/cifar-10-batches-py/' + name)[b'labels'])
            df_data = pd.DataFrame(self.unpickle('blah/cifar-10-batches-py/' + name)[b'data'])
            new = pd.concat([df_labels, df_data], axis=1)
            new.to_csv(name + '.csv', index=False)


CIFAR10_data = CIFAR_10('blah', True, True)
CIFAR10_data.toCSV()
