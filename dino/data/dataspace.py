import sys
import copy
import math
import random
import numpy as np

from enum import Enum
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors

from exlab.interface.serializer import Serializable

from dino.data.data import SingleData, Data, Goal, Action
from .space import Space, SpaceKind

# from ..utils.io import getVisual, plotData, visualize
from . import operations


class DataSpace(Space):
    """
    Abstract class representing a physical space
    """

    def __init__(self, spaceManager, dim, options={}, native=None, kind=SpaceKind.BASIC, spaces=None):
        super().__init__(spaceManager, dim, options=options,
                         native=native, kind=kind, spaces=spaces)

        self._number = 0
        self.clearSpaceWeight()

        self.costs = np.zeros([Space.RESERVE_STEP])
        self.data = np.zeros([Space.RESERVE_STEP, self.dim])
        # execution id (order)
        self.ids = np.zeros([Space.RESERVE_STEP], dtype=np.int32)
        # reverse execution order
        self.lids = np.full([Space.RESERVE_STEP], -1, dtype=np.int32)
        self.actions = []

        self.continuous = True  # ids == lids

    def icon(self):
        return '@☰'

    def canStoreData(self):
        return True

    @property
    def spaceWeights(self):
        return self._spaceWeights

    def _updateSpaceWeights(self):
        if not self._spaceWeights:
            self._nnWeights = np.ones(self.dim) / self.dim
        else:
            weights = np.array(self._spaceWeights)
            weights[:, 1] /= sum(weights[:, 1])
            weights[:, 1] = [weight/space.dim for space, weight in weights]
            self._nnWeights = np.zeros(len(self.spaces))

            def getWeight(space):
                if space not in weights[:, 0]:
                    return 0.
                return [weight for sp, weight in weights if sp == space][0]

            weightPerSpace = [
                [getWeight(space)] * space.dim for space in self.spaces]
            self._nnWeights = np.array(
                [weight for wlist in weightPerSpace for weight in wlist])

    def clearSpaceWeight(self):
        self._spaceWeights = []
        self._updateSpaceWeights()

    def spaceWeight(self, weight, space=None):
        if weight is None:
            self.clearSpaceWeight()
        else:
            if isinstance(weight, list):
                self._spaceWeights = weight
            elif space is None:
                self._spaceWeights = [(space, weight) for space in self.spaces]
            else:
                self._spaceWeights.append((space, weight))
            self._updateSpaceWeights()

    # Data
    @property
    def number(self):
        self._validate()
        return self._number

    def getLid(self, ids):
        self._validate()
        if self.continuous:
            return ids
        a = self.lids[ids]
        return a[a >= 0]

    def getIds(self, restrictionIds=None):
        self._validate()
        return self.ids[:self._number]

    def __getitem__(self, ids):
        self._validate()
        return self.data[self.getLid(ids)]

    def getPoint(self, ids):
        self._validate()
        a = self.lids[ids]
        data = self.data[a[a >= 0]].tolist()
        return [self.point(d) for d in data]

    def getNpPlainPoint(self, ids):
        self._validate()
        a = self.lids[ids]
        return self.data[a[a >= 0]]

    def getPlainPoint(self, ids):
        return self.getNpPlainPoint(ids).tolist()

    def getActionIndex(self, ids):
        self._validate()
        return list(set.intersection(set(ids), set(self.actions)))

    def getNpPlainAction(self, ids):
        ids = self.getActionIndex(ids)
        if not ids:
            return np.array([])
        return self.data[ids]

    def getPlainAction(self, ids):
        list_ = self.getNpPlainAction(ids)
        return [a.tolist() for a in list_]

    # Operations
    def computeDistances(self, x):
        """Compute array of normalized distances between data and the point given."""
        self._validate()
        x = Data.plainData(x, self)
        if self._number == 0:
            return np.array([])
        #return np.sqrt(np.sum(((self.data[:self._number] - x) / self._nnWeights)**2, axis=1)) / self.maxDistance
        return operations._computeDistances(self.data[:self._number], x, self._nnWeights, self.maxDistance)

    def computePerformances(self, x):
        """Compute performances for reaching the given point."""
        return self.computeDistances(x) * self.costs[:self._number]

    def __nnFromData(self, data, n=1, ignore=0, restrictionIds=None, otherSpace=None):
        if self._number == 0:
            return np.array([], dtype=np.int32), np.array([])
        self._validate()
        if otherSpace:
            otherSpace._validate()
        # return self.nnfd(self.lids, self.ids[:self._number], data, n, ignore, restrictionIds, otherSpace)
        return operations._nnFromData(self.lids, self.ids, data, n, ignore, restrictionIds, otherSpace)

    # def getLidfd(self, lids, ids):
    #     a = lids[ids]
    #     return a[a >= 0]
    #
    # def nnfd(self, lids, ids, data, n=1, ignore=0, restrictionIds=None, otherSpace=None):
    #     if otherSpace:
    #         if restrictionIds is not None:
    #             restrictionIds = list(set(restrictionIds).intersection(set(otherSpace.ids[:otherSpace.number])))
    #         else:
    #             restrictionIds = otherSpace.ids[:otherSpace.number]
    #     if restrictionIds is not None:
    #         try:
    #             lids = self.getLidfd(lids, restrictionIds)
    #             data = data[lids]
    #             ids = ids[lids]
    #         except Exception as e:
    #             print(self)
    #             print(lids)
    #             print(self.lids[:100])
    #             print(data[:100])
    #             print(restrictionIds)
    #             raise e
    #
    #     i = data.argpartition(range(ignore, min(n + ignore, data.shape[0])))
    #     i = i[ignore:min(i.size, n)]
    #     return ids[i], data[i]

    def nearest(self, x, n=1, ignore=0, restrictionIds=None, otherSpace=None):
        """Computes Nearest Neighbours based on performances."""
        """For ActionSpaces, nearest and nearestDistance are equivalent, i.e. cost=1"""
        '''return self.__nnFromData(self.computePerformances(x), n=n, ignore=ignore, restrictionIds=restrictionIds,
                                 otherSpace=otherSpace)'''
        return self.__nnFromData(self.computePerformances(x), n=n, ignore=ignore, restrictionIds=restrictionIds,
                                 otherSpace=otherSpace)

    def nearestDistance(self, x, n=1, ignore=0, restrictionIds=None, otherSpace=None):
        """Compute Nearest Neighbours based on distance."""
        return self.__nnFromData(self.computeDistances(x), n=n, ignore=ignore, restrictionIds=restrictionIds,
                                 otherSpace=otherSpace)

    @staticmethod
    def nearestFromData(points, x, n=1, ignore=0):
        return operations._nearestFromData(points, x, n, ignore)

    def nearestDistanceArray(self, x, n=1, ignore=0, restrictionIds=None, otherSpace=None):
        self._validate()
        if otherSpace:
            otherSpace._validate()
        data = self.data[:self._number]
        ids = self.ids[:self._number]

        if otherSpace:
            if restrictionIds is not None:
                restrictionIds = np.array(list(set(restrictionIds).intersection(
                    set(otherSpace.ids[:otherSpace.number]))), dtype=np.int32)
            else:
                restrictionIds = otherSpace.ids[:otherSpace.number]
        if restrictionIds is not None:
            try:
                lids = self.getLid(restrictionIds)
                data = data[lids]
                ids = ids[lids]
            except Exception as e:
                raise e

        if len(ids) == 0:
            return np.array([], dtype=np.int32), np.array([])

        nbrs = NearestNeighbors(n_neighbors=min(
            n + ignore, data.shape[0]), algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(np.array(x))
        return ids[indices[:, ignore:]], distances[:, ignore:]

    def variance(self, ids):
        points = self.getPlainPoint(ids)
        center = np.mean(points, axis=0)
        return np.mean(np.sum((points - center) ** 2, axis=1) ** .5)

    def denseEnough(self, ids, threshold=0.05):
        variance = self.variance(ids)
        return variance < self.maxDistance * threshold

    def getData(self, restrictionIds=None):
        self._validate()
        if restrictionIds is not None:
            try:
                return self.data[self.getLid(restrictionIds)]
            except Exception as e:
                print(self.ids)
                print(restrictionIds)
                raise e
        return self.data[:self._number]

    def getDataSelection(self, n=0, method='first', restrictionIds=None):
        data = self.getData(restrictionIds=restrictionIds)
        if not n:
            n = self.len(data)
        if method == 'random':
            data[np.random.choice(data.shape[0], n)]
        elif method == 'first':
            data = data[:n]
        return data

    def addPoint(self, point, idx, cost=None, action=False):
        """Add a point in the space and if valid return id."""
        if point.space.nativeRoot() != self.nativeRoot():
            return -1

        x = point.plain()

        if self._number > 0 and self.ids[self._number - 1] == idx:
            raise Exception('Trying to add data twice to {} at index {}:\n1: {}\n2: {}'
                            .format(self, idx, self.data[self._number - 1], x))

        # Extend arrays if needed
        if self._number >= self.data.shape[0]:
            self.data = np.append(self.data, np.zeros(
                [Space.RESERVE_STEP, self.dim]), axis=0)
            self.costs = np.append(self.costs, np.zeros(
                [Space.RESERVE_STEP]), axis=0)
            self.ids = np.append(self.ids, np.zeros(
                [Space.RESERVE_STEP], dtype=np.int16), axis=0)
        if idx >= self.lids.shape[0]:
            self.lids = np.append(self.lids, np.full(
                [Space.RESERVE_STEP], -1, dtype=np.int16), axis=0)

        # Store data
        self.data[self._number] = x
        self.costs[self._number] = cost if cost else 1.
        self.ids[self._number] = idx
        self.lids[idx] = self._number
        if idx != self._number:
            self.continuous = False
        self._number += 1
        if action:
            self.actions.append(idx)

        self.invalidate()
        return self._number - 1

    def _postValidate(self):
        if self._number > 0:
            self._bounds = list(zip(np.min(self.data[:self._number], axis=0).tolist(),
                                    np.max(self.data[:self._number], axis=0).tolist()))
            self.maxDistance = math.sqrt(
                sum([(bound[1] - bound[0]) ** 2 for bound in self._bounds]))
        else:
            self.maxDistance = 1.
        self.maxNNDistance = self.maxDistance
        Space._postValidate(self)

    @classmethod
    def getCost(cls, n):
        """Return cost of an action space based on its _number of primitives."""
        return cls.gamma ** n

    # Visual
    def getPointsVisualizer(self, prefix=""):
        """Return a dictionary used to visualize outcomes reached for the specified outcome space."""
        return getVisual(
            [lambda fig, ax, options: plotData(
                self.getData(), fig, ax, options)],
            minimum=[b[0] for b in self._bounds],
            maximum=[b[1] for b in self._bounds],
            title=prefix + "Points in " + str(self)
        )

    def plot(self):
        visualize(self.getPointsVisualizer())

    # Api
    def apiGetPoints(self, ids):
        self._validate()
        # if range_[1] == -1:
        #     ids = np.nonzero(self.ids[:self._number] > range_[0])
        # else:
        #     ids = np.nonzero(np.logical_and(self.ids[:self._number] > range_[0], self.ids[:self._number] <= range_[1]))
        # print(self.ids[ids].size)
        # print(self.data[ids].size)
        # data = np.concatenate((self.ids[ids], self.data[ids]), axis=1)
        return list(zip(self.data[ids].tolist(), self.ids[ids].tolist()))
