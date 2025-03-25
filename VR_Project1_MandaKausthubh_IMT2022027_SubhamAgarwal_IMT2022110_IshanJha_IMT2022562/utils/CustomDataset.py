import torch
from torchvision import transforms
import numpy
import PIL

import os, sys

class CustomDataset(torch.utils.data.Dataset):

    def __init__(self, dataset:torch.Tensor, labels:torch.Tensor, num_classes:int, transform = None, label_transform = None):
        self.num_classes = num_classes
        self.dataset = dataset
        self.transform = transform
        self.label_transform = label_transform
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, label = self.dataset[index], self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.label_transform is not None:
            label = self.label_transform(label)

        return data, label



class DatasetWithPaths(torch.utils.data.Dataset):
    
    def __init__(self, data, data_transform=None, label_transform=None, *args, **kwargs):
        super(DatasetWithPaths, self).__init__(*args, **kwargs)
        self.data = data
        self.data_transform = data_transform
        self.label_transform = label_transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        path, label = self.data[index]
        data = PIL.Image.open(path).convert('RGB')
        if self.data_transform is not None:
            data = self.data_transform(data)
        if self.label_transform is not None:
            label = self.label_transform(label)

        return data, label