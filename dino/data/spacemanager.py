'''
    File name: dataset.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import numpy as np
import math
import random
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean

from exlab.modular.module import Module
from exlab.interface.serializer import Serializable

from dino.representation.entity import Entity

from .data import *
from .space import Space, SpaceKind
from .multispace import MultiColSpace, MultiColDataSpace, MultiRowDataSpace


# import graphviz


class SpaceManager(Module, Serializable):
    def __init__(self, storesData=False, options={}, entityCls=Entity):
        super().__init__()
        self.spaces = []
        self.storesData = storesData
        self.options = options

        self.world = entityCls('root', spaceManager=self)
        self.conserveRoot = False

        self.multiColSpaces = []
        self.multiRowSpaces = []

        self.computeSpaces()

    def _serialize(self, serializer):
        dict_ = {}
        dict_.update(serializer.serialize(
            self, ['spaces', 'storesData'], exportPathType=True))
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        obj = obj if obj else cls(
            dict_.get('storesData'), options=dict_.get('options', {}))
        # Operations
        # spaces = [Space.deserialize(space, obj, options=options)
        #           for space in dict_.get('spaces', [])]
        return obj
    
    def __repr__(self):
        return f'SpaceManager({len(self.spaces)} spaces and {len(self.world.cascadingChildren()) + 1} entities)'
    
    @property
    def size(self):
        return len(self.spaces)

    def registerSpace(self, space):
        if space not in self.spaces:
            self.spaces.append(space)
            self.computeSpaces()

    def computeSpaces(self):
        self.actionSpaces = self.getActionSpaces(self.spaces)
        self.actionExplorationSpaces = self.getActionExplorationSpaces(
            self.spaces)
        self.actionPrimitiveSpaces = self.getActionPrimitiveSpaces(self.spaces)
        self.outcomeSpaces = self.getOutcomeSpaces(self.spaces)

    def _multiSpace(self, spaces, list_, type_):
        # Only 1 space -> return the space itself
        if len(spaces) == 1:
            return spaces[0]

        # Look for exisiting multi space
        r = [s for s in list_ if set(s.spaces) == set(spaces)]
        if len(r) == 0:
            s = type_(self, spaces)
            list_.append(s)
            return s
        else:
            return r[0]

    def _multiSpaceWeighted(self, spaces, list_, type_, weight=None):
        space = self._multiSpace(spaces, list_, type_)

        if weight:
            space.clearSpaceWeight()
            for subspace in spaces:
                space.spaceWeight(weight, subspace)

        return space

    def multiColSpace(self, spaces, canStoreData=None, weight=None):
        # Flatten the list of spaces
        spaces = list(set([subSpace for space in spaces for subSpace in space.cols if space is not None]))

        if len(spaces) == 1:
            return spaces[0]

        if canStoreData is None:
            if spaces:
                dataSpaces = list(set([space.canStoreData()
                                       for space in spaces]))
                if len(dataSpaces) > 1:
                    raise Exception(
                        "All spaces should be a DataSpace or none: {}".format(spaces))
                canStoreData = dataSpaces[0]
            else:
                canStoreData = False

        return self._multiSpaceWeighted(spaces, list_=self.multiColSpaces,
                                        type_=MultiColDataSpace if canStoreData else MultiColSpace, weight=weight)

    def multiRowSpace(self, spaces, canStoreData=None):
        # spaces = list(set([subSpace for space in spaces for subSpace in space]))
        spaces = list(set([space for space in spaces if space is not None]))

        if not spaces:
            return
        if len(spaces) == 1:
            return spaces[0]

        return self._multiSpaceWeighted(spaces, list_=self.multiRowSpaces, type_=MultiRowDataSpace)

    def getActionSpaces(self, spaces):
        return [s for s in spaces if s.controllable() and s.kind == SpaceKind.BASIC]

    def getActionExplorationSpaces(self, spaces):
        return [s for s in spaces if s.controllable() and s.kind == SpaceKind.BASIC
                and not s.noaction]

    def getActionPrimitiveSpaces(self, spaces):
        return [s for s in spaces if s.primitive() and s.kind == SpaceKind.BASIC and not s.noaction]

    def getOutcomeSpaces(self, spaces):
        return [s for s in spaces if s.observable()
                and s.kind == SpaceKind.BASIC]

    def space(self, index_name, kind=SpaceKind.BASIC):
        return next(s for s in self.spaces if (s.name == index_name and (s.kind == kind or s.native == s))
                    or s.id == index_name)

    def spaceSearch(self, property=None, kind=SpaceKind.BASIC):
        if property:
            return ([s for s in self.spaces if s.boundProperty == property and s.kind == kind] + [None])[0]
        return None

    def convertSpace(self, space, kind=None, toData=None):
        toData = toData if toData is not None else self.storesData  # space.canStoreData()
        kind = kind if kind else space.kind

        relatedSpace = ([s for s in self.spaces if s.linkedTo(
            space) and s.kind == kind] + [None])[0]
        if relatedSpace:
            return relatedSpace

        if toData:
            return space.createDataSpace(self, kind)
        else:
            return space.createLinkedSpace(self, kind)

    def convertSpaces(self, spaces, kind=None, toData=None):
        return [self.convertSpace(space, kind=kind, toData=toData) for space in spaces]

    def convertData(self, data, kind=None, toData=None):
        return data.convertTo(self, kind=kind, toData=toData)
    
    def convertEntity(self, entity, proxy=True):
        relatedEntity = ([e for e in self.world.cascadingChildren() if e.linkedTo(entity)] + [None])[0]
        if relatedEntity:
            return relatedEntity

        entity = entity.createLinkedEntity(self, proxy=proxy)
        if entity.isRoot() and not self.conserveRoot:
            self.world = entity
        elif not entity.parent:
            self.world.addChild(entity)
        return entity
