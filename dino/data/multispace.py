import sys
import copy
import math
import random
import numpy as np

from enum import Enum
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors

from exlab.interface.serializer import Serializable

from .data import *
from .space import Space
from .dataspace import DataSpace


class MultiColSpace(Space):
    def __init__(self, spaceManager, spaces, parent=Space):
        dim = sum([space.dim for space in spaces])
        parent.__init__(self, spaceManager, dim, spaces=spaces)

        # Hook when spaces are modified and this one has to be recomputed
        for space in self.spaces:
            space.childrenSpaces.append(self)

    @staticmethod
    def create(spaces, spaceManager=None):
        if not spaces:
            if spaceManager:
                return spaceManager.multiColSpace([])
            return None

        spaceManager = list(set([space.spaceManager for space in spaces]))
        if len(spaceManager) > 1:
            raise Exception(
                "All spaces should lay within the same space manager")
        s = spaceManager[0].multiColSpace(spaces)
        return s

    def icon(self):
        return super().icon() + '∥'

    def boundedProperty(self):
        return '⇉' + '👁' if self.observable() else '' + '🕹' if self.primitive() else ''

    def colStr(self):
        cols = ' '.join([space.toStr(2) for space in self.spaces])
        return f'{self.boundedProperty()}[{cols}]'

    def observable(self):
        # All sub spaces must be observable
        return sum([s.observable() for s in self.spaces]) == len(self.spaces)

    def primitive(self):
        # All sub spaces must be primitive
        return sum([s.primitive() for s in self.spaces]) == len(self.spaces) and not self.null()

    def createDataSpace(self, spaceManager=None, kind=None):
        spaces = [s.convertTo(spaceManager, kind, True) for s in self.spaces]
        return MultiColDataSpace.create(spaces, spaceManager=spaceManager)

    def createLinkedSpace(self, spaceManager=None, kind=None):
        spaces = [s.convertTo(spaceManager, kind, False) for s in self.spaces]
        return MultiColSpace.create(spaces, spaceManager=spaceManager)


class MultiColDataSpace(MultiColSpace, DataSpace):
    def __init__(self, spaceManager, spaces):
        MultiColSpace.__init__(self, spaceManager, spaces, parent=DataSpace)

    def addPoint(self, point, idx, cost=None, action=False):
        """Add a point in the space and if valid return id."""
        # TODO
        if point.space != self:
            return -1

        for part in point.flat():
            part.space.addPoint(part, idx, cost=cost, action=action)

    def invalidate(self):
        Space.invalidate(self)
        self.data = np.array([])
        self.costs = np.array([])
        self.ids = np.array([])
        self.lids = np.array([], dtype=np.int32)
        self.actions = []
        self._number = 0

        '''self.costs = np.zeros([Space.RESERVE_STEP])
        self.data = np.zeros([Space.RESERVE_STEP, self.dim])
        self.ids = np.zeros([Space.RESERVE_STEP], dtype=np.int16)  # execution id (order)
        self.lids = np.full([Space.RESERVE_STEP], -1, dtype=np.int16)  # reverse execution order'''

    def _preValidate(self):
        if not Space._preValidate(self):
            return False
        if not self.spaces:
            return False
        return self._mergeData(col=True)

    def _mergeData(self, col=True):
        for space in self.spaces:
            space._validate()
        if col:
            ids = list(set.intersection(
                *[set(space.ids[:space._number]) for space in self.spaces]))
        else:
            ids = np.arange(np.sum([space.number for space in self.spaces]))
        if len(ids) == len(self.ids) or len(ids) == 0:
            # self.invalid = False
            return True

        # self._bounds = [x for space in self.spaces for x in space._bounds]
        #
        # if len(ids) == len(self.ids):
        #     self.invalid = False
        #     return

        if col:
            self.ids = np.array(ids)
            self.actions = list(set.intersection(
                *[set(space.actions) for space in self.spaces]))
            length = max(
                list(set.union(*[set(space.ids[:space._number]) for space in self.spaces]))) + 1

            def getData(space, length):
                n = np.zeros([length, space.dim])
                '''print(space.data.size)
                print(space.ids.size)
                print(space._number)
                print("HELLLLO")
                print(space.data[:space._number].size)
                print(n[space.ids[:space._number]].size)'''
                n[space.ids[:space._number]] = space.data[:space._number]
                return n

            def getCosts(space, length):
                n = np.zeros([length])
                n[space.ids[:space._number]] = space.costs[:space._number]
                return n

            #print("Compute {} {}".format(length, space._number))
            self.data = np.concatenate(
                [getData(space, length) for space in self.spaces], axis=1)[self.ids]
            self.costs = np.concatenate(
                [getCosts(space, length) for space in self.spaces], axis=0)[self.ids]
            self.lids = np.full([length], -1, dtype=np.int16)
            self.lids[self.ids] = np.arange(len(self.ids))
        else:
            self.ids = ids
            self.actions = np.hstack(
                [space.actions[:space.number] for space in self.spaces])
            self.data = np.vstack([space.data[:space.number]
                                   for space in self.spaces])
            self.costs = np.hstack([space.costs[:space.number]
                                    for space in self.spaces])
            self.lids = np.array(ids, dtype=np.int32)
            # lidsNZero = [np.nonzero(space.lids != -1) for space in self.spaces]
            # self.lids = np.full([np.max(lidsNZero) + 1], -1, dtype=np.int32)
            # pos = 0
            # for space, nz in zip(self.spaces, lidsNZero):
            #     self.lids[nz] = space.lids[nz] + pos
            #     pos += space.number

        self.continuous = all(space.continuous for space in self.spaces)

        #print(self.data)
        self._number = len(self.ids)
        return True

    def _postValidate(self):
        DataSpace._postValidate(self)


class MultiRowDataSpace(DataSpace):
    def __init__(self, spaceManager, spaces):
        assert len(spaces) > 0
        if len(set([space.dim for space in spaces])) != 1:
            raise Exception("All spaces should have the same dimension")
        dim = spaces[0].dim
        super().__init__(spaceManager, dim, native=spaces[0], spaces=spaces)
        self.rowAggregation = True
        self.abstract = True

        # Hook when spaces are modified and this one has to be recomputed
        for space in self.spaces:
            space.childrenSpaces.append(self)

    @staticmethod
    def create(spaces):
        if not spaces:
            return None
        spaceManager = list(set([space.spaceManager for space in spaces]))
        if len(spaceManager) > 1:
            raise Exception(
                "All spaces should lay within the same space manager")
        dataSpaces = list(set([isinstance(space, DataSpace)
                               for space in spaces]))
        if len(dataSpaces) > 1:
            raise Exception("All spaces should be a DataSpace or none")
        return spaceManager[0].multiRowSpace(spaces, data=dataSpaces[0])

    def toStr(self, short=False):
        spaces = '>'.join([space.toStr(2) for space in self.spaces])
        if short:
            return f"({spaces})({self.dim})@⛁"
        return f"@⛁({spaces})({self.dim})"

    # @property
    # def colSpaces(self):
    #     return [self.spaces[0]]

    # @property
    # def colSpacesFull(self):
    #     return [self]

    # @property
    # def rowSpaces(self):
    #     return self.spaces

    @property
    def cols(self):
        return [self]

    @property
    def rows(self):
        return self.spaces

    @property
    def colsType(self):
        return [self.spaces[0]]

    @property
    def rowsType(self):
        return self.spaces

    @property
    def groupedCols(self):
        return [tuple([s for colSpace in space.groupedCols for s in colSpace[0]] for space in self.spaces)]

    def observable(self):
        # All sub spaces must be observable
        return self.spaces[0].observable()

    def primitive(self):
        # All sub spaces must be primitive
        return self.spaces[0].primitive()

    def invalidate(self):
        MultiColDataSpace.invalidate(self)

    def _preValidate(self):
        if not Space._preValidate(self):
            return False
        if len(self.spaces) == 0:
            return False
        return MultiColDataSpace._mergeData(self, col=False)

    def _postValidate(self):
        DataSpace._postValidate(self)
