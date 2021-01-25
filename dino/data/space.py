'''
    File name: space.py
    Author: Alexandre Manoury, Nicolas Duminy
    Python Version: 3.6
'''

import sys
import copy
import math
import random
import logging
import numpy as np

from enum import Enum
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors

from exlab.interface.serializer import Serializable
from exlab.utils.io import parameter

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
        self.rows = self.spaces
        self.aggregation = len(self.spaces) > 1
        self.rowAggregation = False
        self.childrenSpaces = []

        # self.allowedSimilarRows = []

        self.native = native if native else self
        self.nativeRoot = self
        self.nativeRoot = self.native.nativeRoot
        self.kind = kind
        self._property = property

        self.dim = dim
        self.options = options
        # self.delegateto = options.get('delegateto', None)
        self._relative = options.get('relative', False)
        self._relativeLearning = options.get('relativeLearning', True)
        self._modulo = options.get('modulo', None)
        self.noaction = options.get('noaction', False)

        self.abstract = False
        if not self.aggregation:
            self.abstract = any(s.abstract for s in self.spaces)

        self._bounds = [[-1., 1.] for i in range(self.dim)]
        self.maxDistance = 1.
        self.maxDistancePerColumn = np.full(self.dim, 1.)

        # Multi
        self.invalid = False
        self.invalidate()

        # Register
        self.spaceManager = spaceManager
        self.spaceManager.registerSpace(self)
    
    def _sid(self, serializer):
        return serializer.serialize(self, ['kind'], foreigns=['spaceManager', 'boundProperty'], reference=True)

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['kind', 'options'], foreigns=['spaceManager', 'boundProperty'])
        return dict_
    
    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = serializer.get('environment').propertySpace(serializer.deserialize(dict_.get('boundProperty')),
                                                              SpaceKind(dict_.get('kind')),
                                                              serializer.deserialize(dict_.get('spaceManager')))
        return super()._deserialize(dict_, serializer, obj)
    
    # def _postDeserialize(self, dict_, serializer):
    #     super()._postDeserialize(dict_, serializer)

    # @classmethod
    # def _deserialize(cls, dict_, spaceManager, options=None, obj=None):
    #     # Setting gamma
    #     # deprecated
    #     options = dict_.get('options', {})
    #     if 'gamma' in options.keys():
    #         cls.gamma = options['gamma']

    #     # Creating object
    #     spaces = None
    #     obj = obj if obj else cls(spaceManager, dict_.get('dimension', 1), dict_.get('options', {}),
    #                               native=dict_.get('native'), kind=dict_.get('kind', SpaceKind.BASIC),
    #                               spaces=spaces)

    #     # Operations
    #     # Loading results
    #     # if options.get('loadResults') and dict_.get('_number', -1) >= 0:
    #     #     obj.id = dict_.get('id', -1)  # TODO check for id collision id:21
    #     #     obj._number = dict_.get('_number', 0)
    #     #     obj.actions = dict_.get('actions', obj.actions)
    #     #     obj.data = dict_.get('data', obj.data)
    #     #     obj.costs = dict_.get('costs', obj.costs)
    #     #     obj.ids = dict_.get('ids', obj.ids)
    #     #     obj.lids = dict_.get('lids', obj.lids)
    #     return obj

    # Properties
    # def nativeRoot(self):
    #     if self.native == self:
    #         return self
    #     return self.native.nativeRoot()
    
    @property
    def boundProperty(self):
        if self._property:
            return self._property
        return self.nativeRoot._property
    
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
    
    def canStoreData(self):
        return False
    
    @property
    def learnable(self):
        prop = self.boundProperty
        if prop:
            return prop.learnable
        return False

    @property
    def relative(self):
        return self.nativeRoot._relative
    
    @property
    def relativeLearning(self):
        return self.nativeRoot._relativeLearning

    @property
    def modulo(self):
        return self.nativeRoot._modulo

    def null(self):
        return not self.spaces
    
    def __bool__(self):
        return not self.null()

    # Attributes
    @property
    def cols(self):
        return self.spaces

    # @property
    # def rows(self):
    #     return self.spaces + [row for row in self.allowedSimilarRows if row not in self]

    def allowSimilarRows(self, rows):
        self.rows += [row for row in rows if row not in self]
    
    def resetSimilarRows(self):
        self.rows = self.spaces

    @property
    def baseCols(self):
        return self.spaces

    @property
    def baseRows(self):
        return [self.spaces[0]]

    @property
    def flatColsWithoutMultiRows(self):
        # if not self.spaces:
        #     return []
        if not self.aggregation:
            return self.baseCols
        return [colSpace for space in self.baseCols for colSpace in space.flatColsWithoutMultiRows]

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
        return [colSpace for space in self.baseCols for colSpace in space.groupedCols]

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
    
    def findMatchingSpaceRows(self, other, kindSensitive=True, idSensitive=False, entity=None):
        # print(self.flatColsWithMultiRows)
        # print(self.flatColsWithoutMultiRows)
        # print(other.flatColsWithMultiRows)
        # print(other.flatColsWithoutMultiRows)
        for row in self.rows:
            for otherRow in other.rows:
                if row.matches(otherRow, kindSensitive=kindSensitive, idSensitive=idSensitive, entity=entity):
                    return (row, otherRow)
        return None

    def matches(self, other, kindSensitive=True, dataSensitive=False, idSensitive=False, rowMatching=False, entity=None):
        if idSensitive and self != other:
            return False
        if entity and (not self.matchesEntity(entity) or not other.matchesEntity(entity)):
            return False
        if kindSensitive and self.kind.value != other.kind.value:
            return False
        if dataSensitive and self.canStoreData != other.canStoreData:
            return False
        if rowMatching and self.findMatchingSpaceRows(other, kindSensitive=kindSensitive, idSensitive=idSensitive):
            return True
        return self.nativeRoot == other.nativeRoot

    def linkedTo(self, other):
        return self.nativeRoot == other.nativeRoot
    
    def intersects(self, space):
        if isinstance(space, list):
            space = set([sp for s in space for sp in s.flatSpaces])
        else:
            space = set(space.flatSpaces)
        return set(self.flatSpaces).intersection(space)
    
    def includes(self, space):
        return space in self.rows

    # Multi
    def __iter__(self):
        return self.baseCols.__iter__()

    # Points
    def point(self, value, relative=None, toSpace=None):
        d = Data(toSpace if toSpace is not None else self, value)
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

    def plainRandomPoint(self):
        return np.array([random.uniform(minb, maxb) for minb, maxb in self.bounds])

    def asTemplate(self, data, type_item=SingleData, type_vector=Data, entity=None):
        data = list(data)
        if len(data) != self.dim:
            logging.critical(f"Template dimension mismatch: space {self.name} is {self.dim}d and data is {len(data)}d")
        if entity and len(list(self)) == 1:
            parts = [type_item(s, data)
                     for s in self.rows if s.matchesEntity(entity)]
            if not parts:
                parts = [type_item(s, data) for s in self]
        else:
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
    
    def matchesEntity(self, entity):
        if not entity:
            return True
        return self.boundProperty.entity == entity
    
    def applyTo(self, entity):
        if not entity or len(self.rows) == 1:
            return self
        return next((row for row in self.rows if row.matchesEntity(entity)), self)

    # Data
    @property
    def bounds(self):
        self._validate()
        return copy.deepcopy(self._bounds)

    @staticmethod
    def infiniteBounds(dim):
        return [[-math.inf, math.inf] for i in range(dim)]
    
    def convertTo(self, spaceManager=None, kind=None, toData=None):
        spaceManager = parameter(spaceManager, self.spaceManager)
        return spaceManager.convertSpace(self, kind=kind, toData=toData)

    def createLinkedSpace(self, spaceManager=None, kind=None):
        kind = parameter(kind, self.kind)
        spaceManager = spaceManager if spaceManager else self.spaceManager
        return Space(spaceManager, self.dim, native=self.native, kind=kind)

    def createDataSpace(self, spaceManager=None, kind=None):
        from .dataspace import DataSpace
        kind = parameter(kind, self.kind)
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
    
    # Representation
    @property
    def name(self):
        return self.boundProperty.absoluteName if self.boundProperty else None

    def icon(self):
        return '@'

    def colStr(self):
        return self.boundedProperty()
    
    def boundedProperty(self):
        if self._property:
            return f"→{self._property}"
        elif self.boundProperty:
            return f"↝{self.boundProperty}"
        else:
            return "↛"

    def toStr(self, short=False):
        if not self.spaces:
            return '@NullSpace'
        absName = f'#{self.name}' if self.name else ''
        suffix = '' if self.kind.value == SpaceKind.BASIC.value else f':{self.kind.value.upper()}'
        if short == 2:
            return f"{self.boundedProperty()}"
        if short:
            return f"#{self.id}{absName}{self.colStr()}{suffix}↕{self.dim} {self.icon()}"
        return f"{self.icon()}#{self.id}{absName}{self.colStr()}{suffix}↕{self.dim}"

    def __repr__(self):
        return self.toStr()
    
    # Deprecated
    def iterate(self):
        return self.baseCols
