import csv
import pandas

class DataLoader:
    def __init__(self,path='',batchsize=20,shuffling=True,transform=''):
        self.path = path
        self.current_position = 1
        self.batchsize = batchsize
        self.shuffling = shuffling
        self.transform = transform
        with open(self.path) as csv_file:
            self.csv_reader = list(csv.reader(csv_file, delimiter=','))


    def __itr__(self):
        return self

    def __next__(self):
            # df = pandas.read_csv(self.path)  ###############or
            current_data = []
            current_labels = []
            for row in self.csv_reader[self.current_position : self.current_position + self.batchsize]:
                current_data.append(row[1:])
                current_labels.append(row[0])
            self.current_position += self.batchsize
            return current_data, current_labels



