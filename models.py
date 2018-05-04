import time

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm, tqdm_notebook
from sklearn.utils import shuffle

import os

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class ShallowNN(nn.Module):
    def __init__(self):
        super(ShallowNN, self).__init__()
        self.NN = nn.Sequential(
            Flatten(),
            nn.Linear(32*32*3, 2000),
            nn.LeakyReLU(),
            nn.Linear(2000, 1000),
            nn.LeakyReLU(),
            nn.Linear(1000, 500),
            nn.LeakyReLU(),
            nn.Linear(500, 2),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.NN(x), None
        
        
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, padding=1, bias=True)
        self.conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu1 = nn.LeakyReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.LeakyReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu3 = nn.LeakyReLU()
        
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(1024)
        
        self.av1 = nn.AvgPool2d(kernel_size=8)
        self.fc1 = nn.LeakyReLU()
        self.lin = nn.Linear(1024, 2)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.conv12(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.mp1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.mp2(x)
    
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        conv_features=self.conv4(x)
        x=self.bn4(conv_features)
        
        x = self.av1(x)
        x = self.fc1(x.view(x.size(0),-1))
        x = self.lin(x)
        return x, conv_features

