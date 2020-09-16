'''
    File name: loaders.py
    Author: Nicolas Duminy, Alexandre Manoury
    Python Version: 3.6
'''

import scipy.optimize
import numpy as np
import subprocess
import warnings
import logging
import random
import pickle
import json
import time
import csv
import sys
import os

from scipy.optimize import OptimizeResult
from sklearn.linear_model import LinearRegression


from dino.data import operations


def uniformRowSampling(array, probs):
    """
    Choses a row in an array using a weighted random choice
    Exemple:
        uniformRowSampling(['a', 'b'], [2, 1])
        'a' will have twice the chances to appear than 'b'
    """
    probs /= np.sum(probs)
    return array[np.random.choice(len(array), 1, p=probs)[0]]


def uniformSampling(probs):
    """Uniform sampler."""
    probs /= np.sum(probs)
    return np.random.choice(len(probs), 1, p=probs)[0]


def first(list_, default=None):
    if list_:
        return list_[0]
    return default


def iterrange(collection, range_=(-1, -1), key=lambda item: item[0]):
    if isinstance(collection, dict):
        collection = collection.items()
    return list(filter(lambda item: key(item) > range_[0] and (key(item) <= range_[1] or range_[1] == -1), collection))


def popn(list_, number=1, fromEnd=False):
    """
    pops n elements from a list
    """
    ret = []
    for _ in range(number):
        if fromEnd:
            ret.append(list_.pop())
        else:
            ret.append(list_.pop(0))
    return ret


def sigmoid(x):
    """Compute a sigmoid."""
    y = 1.0 / (1.0 + np.exp(-x))
    return y


def thresfunc(value, ratio):
    return (ratio >= 0.5) * (1 - (1 - value) * (0.5 - ratio) * 2) + (ratio < 0.5) * (1 - value * (0.5 - ratio) * 2)


def mixedSort(values1, values2, min1=None, max1=None, min2=None, max2=None):
    if not min1:
        min1 = np.min(values1)
    if not max1:
        max1 = np.max(values1)
    if not min2:
        min2 = np.min(values2)
    if not max2:
        max2 = np.max(values2)
    if min1 < max1:
        values1 = (values1 - min1) / (max1 - min1)
    if min2 < max2:
        values2 = (values2 - min2) / (max2 - min2)

    values = np.array(values1 + values2)
    indices = values.argsort()

    return indices, values


def multivariateRegression(x, y, x0):
    """Compute a multivariate linear regression model y = f(x) using (X,y) and use it to compute f(x0)."""
    try:
        return operations.multivariateRegression(x, y, x0)
    except ValueError as e:
        logging.critical("Regression failed: y is {}x{}d, X is {}x{}d and goal is {}d ({})".format(
            y.shape[0], y.shape[1], x.shape[0], x.shape[1], len(x0), e))


def multivariateRegressionError(x, y, x0, testSetX=None, testSetY=None):
    try:
        return operations.multivariateRegressionError(x, y, x0)
    except ValueError as e:
        logging.critical("Regression failed: y is {}x{}d, X is {}x{}d and goal is {}d ({})".format(
            y.shape[0], y.shape[1], x.shape[0], x.shape[1], len(x0), e))


# def multivariateRegressionVector(X, y, x0):
#     """Compute a multivariate linear regression model y = f(x) using (X,y) and use it to compute f(x0)."""
#     Xf = np.array([x.flatten() for x in X])
#     yf = np.array([x.flatten() for x in y])
#     return y[0].asTemplate(multivariateRegression(Xf, yf, xg.npflatten()).tolist())


def normalEquation(X, y):
    """Compute the multivariate linear regression parameters using normal equation method."""
    theta = np.zeros((X.shape[1], y.shape[1]))
    X2 = X.transpose()
    Xinv = np.linalg.pinv(X2.dot(X))
    theta = Xinv.dot(X2).dot(y)
    return theta
