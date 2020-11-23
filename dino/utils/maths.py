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
from sklearn.linear_model import LinearRegression, RANSACRegressor
from sklearn import linear_model
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


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


def linearValue(start, end, position):
    return start + (end - start) * position


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


def threshold(value, ratio):
    # return (1 - value * (0.5 - ratio) * 2)
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


def multivariateRegression(x, y, x0, columns=None):
    """Compute a multivariate linear regression model y = f(x) using (X,y) and use it to compute f(x0)."""
    if columns is not None:
        x = x[:, columns]
        x0 = np.array(x0)[columns]
    try:
        # ransac = linear_model.RANSACRegressor()
        # ransac.fit(x, y)
        # return ransac.predict([x0])[0]
        return operations.multivariateRegression(x, y, x0)
    except ValueError as e:
        logging.critical(f"Regression failed: y is {y.shape[0]}x{y.shape[1]}d, X is {x.shape[0]}x{x.shape[1]}d and goal is {len(x0)}d ({e})")
        # return operations.multivariateRegression(x, y, x0)


def multivariateRegressionError(x, y, x0, testSetX=None, testSetY=None, columns=None):
    if columns is not None:
        x = x[:, columns]
        x0 = np.array(x0)[columns]
    # if len(x.shape) == 2:
    #     plt.figure()
    #     plt.scatter(x[:, 0], x[:, 1])
    #     plt.scatter(y[:, 0], y[:, 1])
    try:
        # ransac = linear_model.RANSACRegressor()
        # ransac.fit(x, y)
        # # return ransac.predict([x0])[0], 1.-ransac.score(x, y)
        # return ransac.predict([x0])[0], 1.-ransac.score(x[ransac.inlier_mask_], y[ransac.inlier_mask_])

        return operations.multivariateRegressionError(x, y, x0)
    except ValueError as e:
        logging.critical(f"Regression failed: y is {y.shape[0]}x{y.shape[1]}d, X is {x.shape[0]}x{x.shape[1]}d and goal is {len(x0)}d ({e})")
        return operations.multivariateRegressionError(x[:1 + len(x) // 2], y[:1 + len(x) // 2], x0)
    except LinAlgError as e:
        # print('ERRR====')
        # print(x)
        # print(y)
        # print(x0)
        return operations.multivariateRegressionError(x[:1 + len(x) // 2], y[:1 + len(x) // 2], x0)

def mre(x, y, x0):
    # code from scipy LinearRegression:
    number = int(len(x) * 0.9)
    x_offset = np.average(x[:number], axis=0)
    y_offset = np.average(y[:number], axis=0)

    coef_ = np.linalg.lstsq(x[:number] - x_offset, y[:number] - y_offset, rcond=None)[0]
    if y.ndim == 1:
        coef_ = np.ravel(coef_)

    intercept_ = y_offset - np.dot(x_offset, coef_)
    y0 = np.array(x0).dot(coef_) + intercept_
    yp = np.sum(np.square(x.dot(coef_) + intercept_ - y), axis=1)
    error = np.sum(yp)# / np.sum(0.01 + np.square(y - np.mean(y)))
    # print('===')
    # print(x)
    # print(y)
    # print(yp)
    # print(error)
    # + 0.01 to avoid dividing by zero when all y are equal

    return y0, error, yp


def normalEquation(X, y):
    """Compute the multivariate linear regression parameters using normal equation method."""
    theta = np.zeros((X.shape[1], y.shape[1]))
    X2 = X.transpose()
    Xinv = np.linalg.pinv(X2.dot(X))
    theta = Xinv.dot(X2).dot(y)
    return theta
