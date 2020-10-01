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
from exlab.utils.io import parameter

from dino.representation.entity import Entity

from .data import *
from .space import Space, SpaceKind
from .event import InteractionEvent
from .multispace import MultiColSpace, MultiColDataSpace, MultiRowDataSpace


# import graphviz


class SpaceManager(Module, Serializable):
    def __init__(self, storesData=False, options={}, entityCls=Entity, parent=None):
        Module.__init__(self, parent=parent)
        Serializable.__init__(self)
        self.spaces = []
        self.storesData = storesData
        self.options = options

        self.world = entityCls('root', spaceManager=self)
        self.conserveRoot = False

        self.multiColSpaces = []
        self.multiRowSpaces = []

        self.events = {}

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
    
    # Triggered when new data are added
    def updated(self):
        pass

    # Spaces
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
    
    # Space conversion
    def convertSpace(self, space, kind=None, toData=None):
        toData = parameter(toData, self.storesData)  # space.canStoreData()
        kind = parameter(kind, space.kind)

        relatedSpace = next(iter([s for s in self.spaces if s.linkedTo(
            space) and s.kind.value == kind.value]), None)
        if relatedSpace is not None:
            return relatedSpace

        if toData:
            return space.createDataSpace(self, kind)
        else:
            return space.createLinkedSpace(self, kind)

    def convertSpaces(self, spaces, kind=None, toData=None):
        return [self.convertSpace(space, kind=kind, toData=toData) for space in spaces]

    def convertData(self, data, kind=None, toData=None):
        return data.convertTo(self, kind=kind, toData=toData)

    # Multi Spaces
    def _multiSpace(self, spaces, list_, type_, orderWise=False):
        # Only 1 space -> return the space itself
        if len(spaces) == 1:
            return spaces[0]

        # Look for exisiting multi space
        if orderWise:
            r = [s for s in list_ if s.spaces == spaces]
        else:
            r = [s for s in list_ if set(s.spaces) == set(spaces)]
        if len(r) == 0:
            s = type_(self, spaces)
            list_.append(s)
            return s
        else:
            return r[0]

    def _multiSpaceWeighted(self, spaces, list_, type_, weight=None, orderWise=False):
        space = self._multiSpace(spaces, list_, type_, orderWise=orderWise)

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
                        f"All spaces should be a DataSpace or none: {spaces}")
                canStoreData = dataSpaces[0]
            else:
                canStoreData = self.storesData

        return self._multiSpaceWeighted(spaces, list_=self.multiColSpaces,
                                        type_=MultiColDataSpace if canStoreData else MultiColSpace, weight=weight)

    def multiRowSpace(self, spaces, canStoreData=None):
        # spaces = list(set([subSpace for space in spaces for subSpace in space]))
        # spaces = list(set([space for space in spaces if space is not None]))
        spaces = [space for space in spaces if space is not None]

        if not spaces:
            return
        if len(spaces) == 1:
            return spaces[0]

        return self._multiSpaceWeighted(spaces, list_=self.multiRowSpaces, type_=MultiRowDataSpace, orderWise=True)

    # List Spaces
    def space(self, index_name, kind=SpaceKind.BASIC):
        return next(s for s in self.spaces if (s.name == index_name and (s.kind.value == kind.value or s.native == s))
                    or s.id == index_name)

    def spaceSearch(self, property=None, kind=SpaceKind.BASIC):
        if property:
            return next(iter([s for s in self.spaces if s.boundProperty == property and s.kind.value == kind.value]), None)
        return None

    def getActionSpaces(self, spaces):
        return [s for s in spaces if s.controllable() and s.kind == SpaceKind.BASIC]

    def getActionExplorationSpaces(self, spaces):
        return [s for s in spaces if s.controllable() and s.kind == SpaceKind.BASIC
                and not s.noaction]

    def getActionPrimitiveSpaces(self, spaces):
        return [s for s in spaces if s.primitive() and s.kind.value == SpaceKind.BASIC.value and not s.noaction]

    def getOutcomeSpaces(self, spaces):
        return [s for s in spaces if s.observable()
                and s.kind.value == SpaceKind.BASIC.value]

    # Data
    def addEvent(self, event, cost=1.):
        """Add data to the dataset"""
        event = event.clone()
        event.addToSpaces(cost=cost)
        self.logger.debug(f'Adding point {event} to dataset {self}', tag='dataset')
        self.events[event.iteration] = event
        # eventId = self.nextEventId()  # used to identify the execution order
        # self.iterationIds.append([self.iteration, eventId])
        # event.iteration = self.iteration
        # self.iteration += 1
        # Register the iteration when the event was done
        # if self.iteration == event.iteration + 1:
        #     self.iterationIds[-1].append(eventId)
        # elif self.iteration < event.iteration + 1:
        #     self.iterationIds.append([event.iteration, eventId])
        #     self.iteration = event.iteration + 1
        # else:
        #     raise Exception("Adding event in the past is forbidden!")

        '''if event.outcomes.get()[0].value[0] ** 2 + event.outcomes.get()[0].value[1] ** 2 < 0.1:
            if event.actions.get()[0].get()[0].value[0] ** 2 + event.actions.get()[0].get()[0].value[1] ** 2 > 0.05:'''
        #print("{}: {} -> {}".format(eventId, event.actions, event.outcomes))

        # event.id = eventId
        

        # assert a_type[0] < len(self.actionSpaces)
        '''if p_type:
            idP = self.p_spaces[p_type[0]][p_type[1]].addPoint(p, eventId)
            self.idP.append((p_type, idP))
            self.idEP.append(eventId)
        else:
            self.idP.append(None)'''

    def actions(self):
        return [event.actions for event in self.events]

    def eventFromId(self, iteration):
        register = self.events[iteration]
        event = InteractionEvent(iteration)
        event.actions = ActionList(Action(
            *(SingleAction(t, self.getData(t, v).tolist()) for t, v in register.actions)))
        event.outcomes = Observation(
            *(SingleObservation(t, self.getData(t, v).tolist()) for t, v in register.outcomes))
        return event

    def getData(self, space_id, data_id):
        return self.space(space_id).data[data_id]
    
    # Entities
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
