import numpy as np
import pandas as pd
import math

np.random.seed(0)
def read_data(x_train_path,y_train_path,x_test_path):
    X_train,Y_train,X_test = [],[],[]
    with open(x_train_path,'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            X_train.append(line.strip('\n').split(',')[1:])
    with open(y_train_path,'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            Y_train.append(line.strip('\n').split(',')[1])
    with open(x_test_path,'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            X_test.append(line.strip('\n').split(',')[1:])

    X_train = np.array(X_train,dtype=float)
    Y_train = np.array(Y_train,dtype=float)
    X_test = np.array(X_test,dtype=float)
    return X_train,Y_train,X_test


def normalize_data(X,is_train=True,cols=None,X_mean=None,X_std=None):
    '''
    :param X: data need to be normalized
    :param is_train: Is the data training set or not
    :param cols: cols that need to be processed
    :param X_mean:  mean of training data
    :param X_std:  std of training data
    :return: normalized data,data's mean,data's std
    '''
    if cols == None:
        cols = np.arange(X.shape[1])
    if is_train:
        X_mean = np.mean(X[:,cols],axis=0).reshape(1,-1)
        X_std = np.std(X[:,cols],axis=0).reshape(1,-1)

    X[:,cols] = (X[:,cols] - X_mean) / (X_std + 1e-8)
    return X,X_mean,X_std



def split_train_and_valid(X,y,split_ratio = 0.8):
    x_train_set = X[: math.floor(len(X) * split_ratio)]
    y_train_set = y[: math.floor(len(y) * split_ratio)]
    x_validation = X[math.floor(len(X) * split_ratio):]
    y_validation = y[math.floor(len(y) * split_ratio):]
    return x_train_set,y_train_set,x_validation,y_validation


def sigmoid(z):
    '''
    :return: to avoid overflow,constrain value between 1e-8 and 1-1e-8
    '''
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def f(X,w,b):
    return sigmoid(np.matmul(X,w)+b)

def predict(X,w,b):
    return np.round(f(X,w,b)).astype(np.int)

def accruacy(Y_pred,Y_hat):
    return 1 - np.mean(np.abs(Y_pred - Y_hat))

def cross_entropy_loss(y_pred,y_hat):
    return -np.dot(y_hat,np.log(y_pred)) - np.dot((1-y_hat),np.log(1-y_pred))

def gradient_descent(X,Y_hat,w,b):
    y_pred = f(X, w, b)
    pred_error = Y_hat - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad,b_grad

def shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

