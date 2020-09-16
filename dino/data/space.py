'''
    File name: space.py
    Author: Alexandre Manoury, Nicolas Duminy
    Python Version: 3.6
'''

import sys
import copy
import math
import random
import numpy as np

from enum import Enum
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors

from exlab.interface.serializer import Serializable

# from dino.utils.io import getVisual, plotData, visualize
# from dino.utils.logging import Logger
from dino.utils.maths import popn
from dino.data.data import SingleData, Data, Goal, Action


class SpaceKind(Enum):
    # NATIVE = 'native'
    BASIC = 'basic'
    PRE = 'pre'


"""
"""


class FormatParameters(object):
    def __init__(self):
        self.spaces = {}


class Space(Serializable):
    RESERVE_STEP = 10000
    _number = 0

    def __init__(self, spaceManager, dim, options={}, native=None, kind=SpaceKind.BASIC, spaces=None, property=None):
        self.id = Space._number
        Space._number += 1

        self.spaces = spaces if spaces is not None else [self]
        self.aggregation = len(self.spaces) > 1
        self.rowAggregation = False
        self.childrenSpaces = []

        self.native = native if native else self
        self.kind = kind
        self._property = property

        self.dim = dim
        self.options = options
        # self.delegateto = options.get('delegateto', None)
        self._relative = options.get('relative', True)
        self._modulo = options.get('modulo', None)
        self.noaction = options.get('noaction', False)

        self.abstract = False
        if not self.aggregation:
            self.abstract = any(s.abstract for s in self.spaces)

        self._bounds = [[-1., 1.] for i in range(self.dim)]
        self.maxDistance = 1.

        # Multi
        self.invalid = False
        self.invalidate()

        # Register
        self.spaceManager = spaceManager
        self.spaceManager.registerSpace(self)

    # def cid(self):
    #     return self.id

    # def gid(self):
    #     if self.boundProperty:
    #         return Serializer.make_gid(self, self.spaceManager.gid(), self.kind.value, self.boundProperty.gid())
    #     return Serializer.make_gid(self, self.spaceManager.gid(), self.kind.value)

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['id', '_property', 'kind', 'options', 'native'])
        # dict_ = Serializer.serialize(self, ['id', 'kind', 'options', 'native'], options=options)
        return dict_

    @classmethod
    def _deserialize(cls, dict_, spaceManager, options=None, obj=None):
        # Setting gamma
        # deprecated
        options = dict_.get('options', {})
        if 'gamma' in options.keys():
            cls.gamma = options['gamma']

        # Creating object
        spaces = None
        obj = obj if obj else cls(spaceManager, dict_.get('dimension', 1), dict_.get('options', {}),
                                  native=dict_.get('native'), kind=dict_.get('kind', SpaceKind.BASIC),
                                  spaces=spaces)

        # Operations
        # Loading results
        # if options.get('loadResults') and dict_.get('_number', -1) >= 0:
        #     obj.id = dict_.get('id', -1)  # TODO check for id collision id:21
        #     obj._number = dict_.get('_number', 0)
        #     obj.actions = dict_.get('actions', obj.actions)
        #     obj.data = dict_.get('data', obj.data)
        #     obj.costs = dict_.get('costs', obj.costs)
        #     obj.ids = dict_.get('ids', obj.ids)
        #     obj.lids = dict_.get('lids', obj.lids)
        return obj

    def boundedProperty(self):
        if self._property:
            return "→{}".format(self._property)
        elif self.boundProperty:
            return "↝{}".format(self.boundProperty)
        else:
            return "↛"

    @property
    def learnable(self):
        prop = self.boundProperty
        if prop:
            return prop.learnable
        return False

    @property
    def name(self):
        return self.boundProperty.absoluteName if self.boundProperty else None

    def icon(self):
        return '@'

    def colStr(self):
        return self.boundedProperty()

    def toStr(self, short=False):
        if not self.spaces:
            return '@NullSpace'
        absName = '#{}'.format(self.name) if self.name else ''
        suffix = '' if self.kind == SpaceKind.BASIC else ':{}'.format(self.kind.value.upper())
        if short == 2:
            return "{}".format(self.boundedProperty())
        if short:
            return "#{}{}{}{}↕{} {}".format(self.id, absName, self.colStr(), suffix, self.dim, self.icon())
        return "{}#{}{}{}{}↕{}".format(self.icon(), self.id, absName, self.colStr(), suffix, self.dim)

    def __repr__(self):
        return self.toStr()

    def convertTo(self, spaceManager=None, kind=None, toData=None):
        spaceManager = spaceManager if spaceManager else self.spaceManager
        return spaceManager.convertSpace(self, kind=kind, toData=toData)

    @property
    def cols(self):
        return self.spaces

    @property
    def rows(self):
        return [self]

    @property
    def colsType(self):
        return self.spaces

    @property
    def rowsType(self):
        return [self.spaces[0]]

    @property
    def flatCols(self):
        # if not self.spaces:
        #     return []
        if not self.aggregation:
            return self.cols
        return [colSpace for space in self.colsType for colSpace in space.flatCols]

    @property
    def flatColsWithMultiRows(self):
        if len(self.cols) == 1 and self.cols[0] == self:
            return self.cols
        return [colSpace for space in self.cols for colSpace in space.flatColsWithMultiRows]

    @property
    def flatSpaces(self):
        if not self.aggregation:
            return self.spaces
        return [colSpace for space in self.spaces for colSpace in space.flatSpaces]

    @property
    def groupedCols(self):
        if not self.aggregation:
            return [([self],)]
        return [colSpace for space in self.colsType for colSpace in space.groupedCols]

    # @property
    # # List all col-stacked (Vertical Stack) spaces in the current space
    # def colSpaces(self):
    #     return self.spaces

    # @property
    # def colSpacesAll(self):
    #     if not self.spaces:
    #         return []
    #     if self.spaces[0] == self:
    #         return self.spaces
    #     return [colSpace for space in self.spaces for colSpace in space.colSpacesAll]

    # @property
    # def colSpacesFull(self):
    #     return self.spaces

    # @property
    # # List all row-stacked (Horizontal Stack) spaces
    # def rowSpaces(self):
    #     return [self.spaces[0]]

    def columnsFor(self, space):
        pos = 0
        colIds = []
        for s in self.cols:
            if s.intersects(space):
                colIds += range(pos, pos + s.dim)
            pos += s.dim
        return colIds

    def nativeRoot(self):
        if self.native == self:
            return self
        return self.native.nativeRoot()

    def matches(self, other, kindSensitive=True, dataSensitive=False):
        if kindSensitive and self.kind != other.kind:
            return False
        if dataSensitive and self.canStoreData != other.canStoreData:
            return False
        return self.nativeRoot() == other.nativeRoot()

    def canStoreData(self):
        return False

    @property
    def boundProperty(self):
        if self._property:
            return self._property
        return self.nativeRoot()._property

    @property
    def relative(self):
        return self.nativeRoot()._relative

    @property
    def modulo(self):
        return self.nativeRoot()._modulo

    def null(self):
        return not self.spaces

    def observable(self):
        if self.boundProperty is None or self.null():
            return False
        return self.boundProperty.observable()

    def controllable(self):
        return True

    def primitive(self):
        if self.boundProperty is None or self.null():
            return False
        return self.boundProperty.controllable()

    def linkedTo(self, otherSpace):
        return self.nativeRoot() == otherSpace.nativeRoot()

    # Multi
    def __iter__(self):
        return self.colsType.__iter__()

    # Deprecated
    def iterate(self):
        return self.colsType

    def __bool__(self):
        return len(self.spaces) > 0

    # Points
    def point(self, value, relative=None):
        d = Data(self, value)
        d.setRelative(relative)
        return d

    def goal(self, value, relative=None):
        d = Goal(self, value)
        d.setRelative(relative)
        return d

    def action(self, value, relative=None):
        d = Action(self, value)
        d.setRelative(relative)
        return d

    def zero(self, relative=None):
        self._validate()
        d = Data(self, [0] * self.dim)
        d.setRelative(relative)
        return d

    def plainZero(self):
        self._validate()
        return np.array([0.] * self.dim)

    def intersects(self, space):
        return set(self.flatSpaces).intersection(set(space.flatSpaces))

    def plainRandomPoint(self):
        return np.array([random.uniform(minb, maxb) for minb, maxb in self.bounds])

    def asTemplate(self, data, type_item=SingleData, type_vector=Data):
        data = list(data)
        if len(data) != self.dim:
            Logger.main().critical("Template dimension mismatch: space {} is {}d and data is {}d".format(
                         self.name, self.dim, len(data)))
        parts = [type_item(s, popn(data, s.dim)) for s in self]
        return type_vector(*parts)

    def formatData(self, data, formatParameters=None):
        if type(data) is not np.ndarray:
            data = np.array(data)
        if formatParameters:
            if self.native not in formatParameters.spaces:
                formatParameters.spaces[self.native] = {}
            settings = formatParameters.spaces[self.native]
        else:
            settings = {}

        if self.modulo is not None:
            settings['modulo'] = settings.get('modulo', data // self.modulo)
            data = data - settings['modulo'] * self.modulo

        return data

    # Data
    @property
    def bounds(self):
        self._validate()
        return copy.deepcopy(self._bounds)

    def createLinkedSpace(self, spaceManager=None, kind=None):
        kind = kind if kind else self.kind
        spaceManager = spaceManager if spaceManager else self.spaceManager
        return Space(spaceManager, self.dim, native=self.native, kind=kind)

    def createDataSpace(self, spaceManager=None, kind=None):
        from .dataspace import DataSpace
        kind = kind if kind else self.kind
        spaceManager = spaceManager if spaceManager else self.spaceManager
        return DataSpace(spaceManager, self.dim, native=self.native, kind=kind)

    # Validation
    def invalidate(self):
        if self.invalid:
            return

        self.invalid = True
        for space in self.childrenSpaces:
            space.invalidate()

    def _validate(self):
        if self._preValidate():
            self._postValidate()

    def _preValidate(self):
        if not self.invalid:
            return False
        if len(self.spaces) > 1:
            for space in self.spaces:
                space._validate()
        return True

    def _postValidate(self):
        self.invalid = False

    @staticmethod
    def infiniteBounds(dim):
        return [[-math.inf, math.inf] for i in range(dim)]
