import bz2
import sys
import copy
import math
import pickle
import random
import numpy as np

from enum import Enum
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors

from exlab.interface.serializer import Serializable
from exlab.interface.graph import Graph

from dinos.data.data import SingleData, Data, Goal, Action
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

        self.contiguous = True  # ids == lids
        self.incoherentRowsAlignement = False

        self._nnWeightsCache = {}
    
    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(
            self, ['_number', 'contiguous', 'incoherentRowsAlignement']))
        for attr in ['costs', 'data', 'ids']:
            dict_[attr] = bz2.compress(pickle.dumps(getattr(self, attr)[:self._number]))
        a = np.argwhere([self.lids != -1])
        if len(a) == 0:
            n = 0
        else:
            n = np.max(a[:, 1])
        dict_['lids'] = bz2.compress(pickle.dumps(self.lids[:n+1]))
        return dict_
    
    # @classmethod
    # def _deserialize(cls, dict_, serializer, obj=None):
    #     if obj is None:
    #         obj = serializer.get('environment').propertySpace(serializer.get(dict_.get('boundProperty')['__id__']),
    #                                                           SpaceKind(dict_.get('kind')),
    #                                                           serializer.get('dataset'))
    #     return super()._deserialize(dict_, serializer, obj)

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        self._number = dict_.get('number')
        self.contiguous = dict_.get('contiguous')
        self.incoherentRowsAlignement = dict_.get('incoherentRowsAlignement')

        for attr in ['costs', 'data', 'ids', 'lids']:
            if attr in dict_:
                d = pickle.loads(bz2.decompress(dict_.get(attr)))
                getattr(self, attr)[:len(d)] = d
        
        self._validate()
        self._updateSpaceWeights()

    def icon(self):
        return '@☰'

    def canStoreData(self):
        return True

    @property
    def spaceWeights(self):
        return self._spaceWeights

    def _updateSpaceWeights(self):
        if not self._spaceWeights:
            self._nnWeights = np.ones(self.dim, dtype=np.float) / self.dim
        else:
            weights = np.array(self._spaceWeights)
            weights[:, 1] *= sum([space.maxDistance for space in weights[:, 0]]) / sum(weights[:, 1])
            weights[:, 1] = [weight / space.maxDistance / space.dim for space, weight in weights]
            self._nnWeights = np.zeros(len(self.spaces), dtype=np.float)

            def getWeight(space):
                if space not in weights[:, 0]:
                    return 0.
                return [weight for sp, weight in weights if sp == space][0]

            weightPerSpace = [
                [getWeight(space)] * space.dim for space in self.spaces]
            self._nnWeights = np.array(
                [weight for wlist in weightPerSpace for weight in wlist], dtype=np.float)

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
                weight /= len(space.spaces)
                for subspace in space:
                    self._spaceWeights.append((subspace, weight))
            self._updateSpaceWeights()

    # Data
    @property
    def number(self):
        self._validate()
        return self._number

    def getLid(self, ids):
        self._validate()
        if self.contiguous:
            return ids
        a = self.lids[ids]
        return a[a >= 0]

    def getIds(self, restrictionIds=None):
        self._validate()
        return self.ids[:self._number]

    def __getitem__(self, ids):
        self._validate()
        return self.data[self.getLid(ids)]

    def getPoint(self, ids, toSpace=None):
        self._validate()
        a = self.lids[ids]
        data = self.data[a[a >= 0]].tolist()
        return [self.point(d, toSpace=toSpace) for d in data]

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
    def findWeights(self, columns=None, adjustSpaceWeight=None, adjustWeightFactor=1., weights=None):
        # weights = np.array(self._nnWeights) if weights is None else self._nnWeights * weights
        if adjustSpaceWeight and columns is not None and np.any(columns):
            key = (adjustSpaceWeight, adjustWeightFactor, tuple(columns))
            if key in self._nnWeightsCache:
                newWeights = self._nnWeightsCache[key] * self._nnWeights
            else:
                newWeights = np.ones(self.dim)
                adjustColumns = self.columnsFor(adjustSpaceWeight)
                if np.any(adjustColumns):
                    newWeights[adjustColumns] *= adjustWeightFactor * len(adjustColumns) / np.sum(columns[adjustColumns])
                self._nnWeightsCache[key] = newWeights
                newWeights = newWeights * self._nnWeights
        else:
            newWeights = self._nnWeights
        return newWeights if weights is None else newWeights * weights
    
    # def findWeights(self, columns=None, adjustSpaceWeight=None, adjustWeightFactor=1., weights=None):
    #     weights = np.array(self._nnWeights) if weights is None else self._nnWeights * weights
    #     if adjustSpaceWeight and columns is not None and np.any(columns):
    #         key = (adjustSpaceWeight, adjustWeightFactor, columns)
    #         if key in self._nnWeightsCache:
    #             return self._nnWeightsCache[key]
    #         adjustColumns = self.columnsFor(adjustSpaceWeight)
    #         if np.any(adjustColumns):
    #             weights[adjustColumns] *= adjustWeightFactor * len(adjustColumns) / np.sum(columns[adjustColumns])
    #     return weights

    def __computeDistances(self, x, weights=None, columns=None, restrictionLids=None, adjustSpaceWeight=None, adjustWeightFactor=1.):
        """Compute array of normalized distances between data and the point given."""
        data = self.data[:self._number]
        if restrictionLids is not None:
            data = data[restrictionLids]
        weights = self.findWeights(columns, adjustSpaceWeight, adjustWeightFactor, weights)
        # print(data)
        # print(weights)
        try:
            return operations.euclidean_distances(data, x, weights, self.maxDistance, columns)
        except Exception as e:
            raise Exception(f'Failure __computeDistances: {data.dtype} {x.dtype} {weights.dtype}\n{self} {x} {weights} {self.maxDistance} {columns}\n{e}')

    def __computePerformances(self, x, weights=None, columns=None, restrictionLids=None, adjustSpaceWeight=None, adjustWeightFactor=1.):
        """Compute performances for reaching the given point."""
        costs = self.costs[:self._number]
        if restrictionLids is not None:
            costs = costs[restrictionLids]
        return self.__computeDistances(x, weights, columns, restrictionLids, adjustSpaceWeight, adjustWeightFactor) * costs

    def __nearestNeighbors(self, x, n=1, ignore=0, weights=None, columns=None, restrictionIds=None, otherSpace=None,
                           useDistances=True, adjustSpaceWeight=None, adjustWeightFactor=1.):
        self._validate()
        if self._number == 0:
            return np.array([], dtype=np.int32), np.array([])
        if otherSpace:
            otherSpace._validate()
            if otherSpace._number == 0:
                return np.array([], dtype=np.int32), np.array([])

        restrictionLids = operations.findRestrictionsLids(restrictionIds, self, otherSpace)
        if useDistances:
            data = self.__computeDistances(x, weights, columns, restrictionLids, adjustSpaceWeight, adjustWeightFactor)
        else:
            data = self.__computePerformances(x, weights, columns, restrictionLids, adjustSpaceWeight, adjustWeightFactor)
        return operations.nearestNeighbors(self.ids, data, n, ignore, restrictionLids, otherSpace)

    def nearest(self, x, n=1, ignore=0, restrictionIds=None, otherSpace=None, weights=None, columns=None, adjustSpaceWeight=None, adjustWeightFactor=1.):
        """Computes Nearest Neighbours based on performances."""
        return self.__nearestNeighbors(x.npPlain(), n=n, ignore=ignore, weights=weights, columns=columns, restrictionIds=restrictionIds,
                                       otherSpace=otherSpace, adjustSpaceWeight=adjustSpaceWeight, adjustWeightFactor=adjustWeightFactor, useDistances=False)

    def nearestDistance(self, x, n=1, ignore=0, restrictionIds=None, otherSpace=None, weights=None, columns=None, adjustSpaceWeight=None, adjustWeightFactor=1.):
        """Compute Nearest Neighbours based on distance."""
        return self.__nearestNeighbors(x.npPlain(), n=n, ignore=ignore, weights=weights, columns=columns, restrictionIds=restrictionIds,
                                       otherSpace=otherSpace, adjustSpaceWeight=adjustSpaceWeight, adjustWeightFactor=adjustWeightFactor)

    def nearestPlain(self, x, n=1, ignore=0, restrictionIds=None, otherSpace=None, weights=None, columns=None, adjustSpaceWeight=None, adjustWeightFactor=1.):
        """Computes Nearest Neighbours based on performances."""
        return self.__nearestNeighbors(x, n=n, ignore=ignore, weights=weights, columns=columns, restrictionIds=restrictionIds,
                                       otherSpace=otherSpace, adjustSpaceWeight=adjustSpaceWeight, adjustWeightFactor=adjustWeightFactor, useDistances=False)

    def nearestDistancePlain(self, x, n=1, ignore=0, restrictionIds=None, otherSpace=None, weights=None, columns=None, adjustSpaceWeight=None, adjustWeightFactor=1.):
        """Compute Nearest Neighbours based on distance."""
        return self.__nearestNeighbors(x, n=n, ignore=ignore, weights=weights, columns=columns, restrictionIds=restrictionIds,
                                       otherSpace=otherSpace, adjustSpaceWeight=adjustSpaceWeight, adjustWeightFactor=adjustWeightFactor)

    @staticmethod
    def nearestFromData(points, x, n=1, ignore=0):
        return operations.nearestNeighborsFromData(points, x, n, ignore)
    
    @staticmethod
    def nearestFromDataContiguous(points, x, n=1, ignore=0, columns=None):
        return operations.nearestNeighborsFromDataContiguous(points, x, n, ignore, columns)

    def nearestDistanceArray(self, x, n=1, ignore=0, restrictionIds=None, otherSpace=None, weights=None, columns=None):
        self._validate()
        if self._number == 0:
            return np.array([], dtype=np.int32), np.array([])
        if otherSpace:
            otherSpace._validate()
            if otherSpace._number == 0:
                return np.array([], dtype=np.int32), np.array([])

        weights = self.maxDistancePerColumn
        data = self.data[:self._number]
        ids = self.ids[:self._number]

        restrictionLids = operations.findRestrictionsLids(restrictionIds, self, otherSpace)
        if restrictionLids is not None:
            data = data[restrictionLids]
            ids = ids[restrictionLids]

        if len(ids) == 0:
            return np.array([], dtype=np.int32), np.array([])

        x = np.array(x)
        weights = self._nnWeights if weights is None else self._nnWeights * weights
        if columns is not None:
            data = data[:, columns]
            x = x[:, columns]
            weights = weights[columns]
        data = data * weights
        x = x * weights

        nbrs = NearestNeighbors(n_neighbors=min(
            n + ignore, data.shape[0]), algorithm='ball_tree').fit(data)
        distances, indices = nbrs.kneighbors(x)
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
            n = len(data)
        if method == 'random':
            data[np.random.choice(data.shape[0], n)]
        elif method == 'first':
            data = data[:n]
        return data
    
    # def findDuplicates(self, starting=-1):
    #     self._validate()
    #     if self.lastDuplicateSearch >= self._number:
    #         return

    #     self.noDuplicates = 

    #     self.lastDuplicateSearch = self._number

    def addPoint(self, point, idx, cost=None, action=False):
        """Add a point in the space and if valid return id."""
        if point.space.nativeRoot != self.nativeRoot:
            return -1

        x = point.plain()

        if self._number > 0 and self.ids[self._number - 1] == idx:
            raise Exception(f'Trying to add data twice to {self} at index {idx}:\n1: {self.data[self._number - 1]}\n2: {x}')

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
            self.contiguous = False
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
        if self.maxDistance == 0:
            self.maxDistance = 1.
        if self.aggregation:
            for space in self.spaces:
                self.maxDistancePerColumn[self.columnsFor(space)] = space.maxDistance
        else:
            self.maxDistancePerColumn = np.full(self.dim, self.maxDistance)
        self.maxNNDistance = self.maxDistance
        Space._postValidate(self)

    # @classmethod
    # def getCost(cls, n):
    #     """Return cost of an action space based on its _number of primitives."""
    #     return cls.gamma ** n

    # Visual
    def visualizeData(self, options={}):
        g = Graph(title=f'Points from {self}', options=options)
        g.scatter(self.getData())
        return g

    # def getPointsVisualizer(self, prefix=""):
    #     """Return a dictionary used to visualize outcomes reached for the specified outcome space."""
    #     return getVisual(
    #         [lambda fig, ax, options: plotData(
    #             self.getData(), fig, ax, options)],
    #         minimum=[b[0] for b in self._bounds],
    #         maximum=[b[1] for b in self._bounds],
    #         title=prefix + "Points in " + str(self)
    #     )

    # def plot(self):
    #     visualize(self.getPointsVisualizer())

    # # Api
    # def apiGetPoints(self, ids):
    #     self._validate()
    #     # if range_[1] == -1:
    #     #     ids = np.nonzero(self.ids[:self._number] > range_[0])
    #     # else:
    #     #     ids = np.nonzero(np.logical_and(self.ids[:self._number] > range_[0], self.ids[:self._number] <= range_[1]))
    #     # print(self.ids[ids].size)
    #     # print(self.data[ids].size)
    #     # data = np.concatenate((self.ids[ids], self.data[ids]), axis=1)
    #     return list(zip(self.data[ids].tolist(), self.ids[ids].tolist()))
