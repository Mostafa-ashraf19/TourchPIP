import os
import requests
import zipfile, urllib.request, shutil


url = 'https://storage.googleapis.com/kaggle-competitions-data/kaggle-v2/3004/861823/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1611346866&Signature=nfS1R4EEvEym8%2FE5TF41PTEKmeHV0yRm%2BZ59igadyi6jKW5Pq620gMRJq7ZVeqPUNeT%2BVslHBms07g47Lj%2FmucOyAl37Hve0l3AtwauAgIDunduzKLfQ7qgp7TyLvNo3DsDRBSJ%2BiYf%2BZ7PIZ6UOArE7p1T1JUROvwejzYRjK8VfAaRH0VXEU%2BFFQUVXcdN8lxmYjFJbNFzy2Wv5m%2FENn6c9SN%2Fr6J1g4W9QnSukKHAhUnLvS1wSdO8RHtnbR27ofaDP4%2B3h0VE6LVRVtmNi%2Bj04K8bOERUKtdfgHeKIXRXlh2ZdZRM8DYoOaloVl2cYtCRyD5IFSxheUP0fbv3HnQ%3D%3D&response-content-disposition=attachment%3B+filename%3Ddigit-recognizer.zip'
file_name = 'myzip.zip'


class FashionMNIST :
    def __init__(self,path,download=True,train=True):
        self.path = path
        self.download = download
        self.train = train
        if self.download:
            self._Download()
        self.path = os.getcwd() + '/' + self.path + '/test.csv' if not self.train else os.getcwd() + '/' + self.path +'/train.csv'
    
    def _Download(self):
        if not os.path.exists(os.getcwd() + '/' + self.path):
            os.mkdir(self.path)
            
            with urllib.request.urlopen(url) as response, open(os.getcwd() + '/' + self.path +'/'+ file_name, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
                with zipfile.ZipFile(os.getcwd() + '/' + self.path + '/' +file_name) as zf:
                    zf.extractall(os.getcwd() + '/' + self.path + '/')
        
    
    # def data(self):
    #     self._Download()
    #     if not self.train:
    #         self.path = os.getcwd() + '/' + self.path + '/test.csv'
    #         return 
    #     return os.getcwd() + '/' + self.path +'/train.csv'
        

