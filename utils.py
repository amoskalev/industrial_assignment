import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from skimage import img_as_float
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm, tqdm_notebook
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import os

def read_folder(path='clock/'):
    """Read images from specifed folder.
    
    Args:
        path: folder
        
    Returns:
        Returns an array with each image as np.floar32 CxWxH 
    """
    out = list()
    for each_img in os.listdir(path):
        out.append(np.transpose(img_as_float(imread(path + each_img)), (2,0,1)))
    return np.array(out)

def train_test_val_split(data_X, data_y, test_ratio = 0.3, val_ratio = 0.2):
    """Split given data to train, test, validation
    
    Args:
        data_X: data to be splitted
        data_y: respective labels to be splitted
        test_ratio: defines what ratio of the whole dataset will be used for train/test+validation division
        val_ratio: defines what ratio of the test+validation data will be given to validation data
        
    Returns:
        X_train, y_train, X_test, y_test, X_val, y_val
    """
    
    X_train, X_val_, y_train, y_val_ = train_test_split(data_X, data_y, test_size = test_ratio) 
    X_val, X_test, y_val, y_test = train_test_split(X_val_, y_val_, test_size = 1 - val_ratio)
    
    print('Train size:',X_train.shape)
    print('Test size:',X_test.shape)
    print('Validation size:',X_val.shape)
    
    return X_train, y_train, X_test, y_test, X_val, y_val

def augment_rotation(X,dummy=None):
    """Flip images in an array.
    
    Args:
        X: array of images
        dummy: the wooden crutch
        
    Returns:
        Returns an array with each image flipped
    """
    return np.flip(X, axis=3)

def augment_crop(X, size=24):
    """Does random cropp of an image. Assume images are squares
    
    Args:
        X: an image to be cropped
        size: the size of cropped frame
        
    Returns:
        Returns a cropped image rescaled to the initial size
    """
    d = X.shape[2] - size
    dx = np.random.randint(X.shape[1]-d, X.shape[1])
    dy = np.random.randint(X.shape[2]-d, X.shape[2])
    cropped = X[:,(dx-size):dx,(dy-size):dy].copy()
    if np.random.rand() > 0.5:
        return np.flip(resize(cropped, output_shape=X.shape, mode='reflect'),2)
    else:
        return resize(cropped, output_shape=X.shape, mode='reflect')

def augment_dataset(X_data, y_data, n=2):
    """Does flips and crops on the dataset
    
    Args:
        X_data: array of images
        y_data: array of labels
        n: if n=1 then will augment each image, if 2 then will augment each second image, etc.
        
    Returns:
        imput array of images concatenated with augmented images, labels
    """
    rotated = np.apply_over_axes(func=augment_rotation, a=X_data, axes=[0])
    
    X_cropped, y_cropped = list(), list()
    for i in tqdm(range(0,X_data.shape[0],n)):
        X_cropped.append(augment_crop(X_data[i]))
        y_cropped.append(y_data[i])
    
    X_cropped, y_cropped = np.array(X_cropped), np.array(y_cropped)
    
    augmented_X = np.concatenate((X_data,rotated,X_cropped))
    augmented_y = np.concatenate((y_data,y_data,y_cropped))
    print('Augmented Train size:',augmented_X.shape)
    return augmented_X, augmented_y

def iterate_minibatches(X, y, batchsize):
    """Generate minibatches
    
    Args:
        X: data
        y: target
        batchsize: how many images will be in a returned batch
        
    Returns:
        Returns a batch of specified size
    """
    indices = np.random.permutation(np.arange(len(X)))
    for start in range(0, len(indices), batchsize):
        ix = indices[start: start + batchsize]
        yield X[ix], y[ix]
        
def compute_loss(model, X_batch, y_batch):
    """Compute cross-entropy loss for training 
    
    Args:
        model: NN model to train
        X_batch: data batch
        y_batch: batch of targets
        
    Returns:
        loss, convolutional features
    """
    X_batch = Variable(torch.FloatTensor(X_batch).cuda())
    y_batch = Variable(torch.LongTensor(y_batch).cuda())
    logits, img_features = model(X_batch)
    return F.cross_entropy(logits, y_batch).mean(), img_features

def get_perceptual_projections(projection_model, X_data):
    """Grab features that neural network produces
    
    Args:
        projection_model: NN model that returns (prediction, convolutional features)
        X_batch: data batch
        y_batch: batch of targets
        
    Returns:
        loss, convolutional features
    """
    print("Estimating convolutional features...")
    projection_model.train(False)
    conv_list = list()
    for X_batch in X_data:
        _,conv_features = projection_model(Variable(torch.FloatTensor(X_batch[None,...])).cuda())
        conv_list.append(conv_features.data.cpu().numpy()[0,...])
    return np.array(conv_list)

def get_perceptual_neighbors(trained_model, first_cluster, second_cluster, N=15):
    """Grab features that neural network produces
    
    Args:
        trained_model: NN model that produces convolutional features as (prediction, convolutional features)
        first_cluster: data that belongs to the first class
        second_cluster: data that belongs to the second class
        N: number of images to return
        
    Returns:
        an array of N images
    """
    
    crocs_conv_norm = np.mean(get_perceptual_projections(trained_model, first_cluster),0)
    clock_conv_norm = np.mean(get_perceptual_projections(trained_model, second_cluster),0)
    
    X_data = np.concatenate([second_cluster.copy(), first_cluster.copy()], axis=0)
    X_conv = get_perceptual_projections(trained_model, X_data)
    
    perceptual_neighbors = X_conv + clock_conv_norm + crocs_conv_norm
    
    pen = []
    for each_el in perceptual_neighbors:
        pen.append(np.linalg.norm(each_el))
        
    pen = np.argsort(np.array(pen))[::-1]
    pen = shuffle(pen[500-N//2+1:500+(N-N//2+1)])
    
    return X_data[pen]