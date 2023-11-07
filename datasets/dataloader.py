from torch.utils.data import Dataset, DataLoader
import os
from scipy.io import loadmat
import numpy as np
import torch
import h5py

class MyDataset(Dataset):
    def __init__(self, rest, label, Tlabel = 0):
        restlist = os.listdir(rest)
        self.Tlabel = Tlabel
        #print(restlist)
        if self.Tlabel == 0:
            restlist = [item for item in restlist if int(str(item).split('.')[0]) <= 600]
        elif self.Tlabel == 1:
            restlist = [item for item in restlist if int(str(item).split('.')[0]) > 600]
        #print(restlist)
        self.rest = [os.path.join(rest, item) for item in restlist]
        self.data = self.rest
        self.label = label
        # self.data=self.motor+self.rest

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        image = h5py.File(image)
        image = image.get('fc_matrix')
        image = np.array(image)[:136,:136]
        label = torch.tensor(self.label)
        return image, label



'''
data_dir = str(os.path.abspath(os.path.join(os.getcwd(), ".")) + '/Data/')

Rest1 = 'REST1/'
train_dataset = MyDataset(data_dir+Rest1, 1, 0)
print(train_dataset)
print(train_dataset)

train_dataloader = DataLoader(train_dataset, 1)

for ii, jj in train_dataloader:
    print(ii)
    print(jj)
    break
'''
