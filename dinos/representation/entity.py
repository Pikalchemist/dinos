'''
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import numpy as np

from exlab.interface.serializer import Serializable, Serializer
from exlab.utils.text import colorText, Colors
from exlab.utils.ensemble import Ensemble
from exlab.utils.io import parameter

from dinos.data.data import Observation
from dinos.data.space import SpaceKind


class Entity(Serializable):
    """Represents a world entity with a set of properties (features and effectors)

    Args:
        kind (string): the type of our entity
        absoluteName (string): Absolute name used to point to this entity. Unique

    Attributes:
        index (int): Absolute index
        indexKind (int): Index for the given entity type
        parent (Entity):
        activated (bool):
        absoluteName
        kind

    """

    number = 0
    indexes = {}

    def __init__(self, kind, absoluteName='', disconnected=False, manager=None):
        self.absoluteName = absoluteName
        self.kind = kind

        # self.proxy = proxy

        self._manager = manager

        self.disconnected = disconnected
        self._discretizeStates = False
        self._discretizeActions = False

        self.native = self

        # if proxy:
        #     native = proxy
        # self.native = native if native else self

        # Indexing
        self.index = Entity.number
        self.indexKind = Entity.indexes.get(self.kind, 0)
        Entity.number += 1
        Entity.indexes[self.kind] = self.indexKind + 1

        self._properties = {}
        self._children = []
        self.parent = None
        self.activated = False

        self.filterObservables = None

    def _sid(self, serializer, raw=False):
        if self.absoluteName:
            sid = f'#{self.absoluteName}'
        else:
            sid = f'{self.kind}:{self.indexKind}'
        if raw:
            return sid
        else:
            return serializer.uid('entity', sid)

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['kind', 'absoluteName', 'index', 'indexKind', 'parent', '_children', '_properties'])
        return dict_

    # Links and proxies
    def linkedTo(self, entity):
        return self.native == entity.native

    def convertTo(self, manager=None):
        manager = parameter(manager, self.manager)
        return manager.convertEntity(self)
    
    def createLinkedEntity(self, manager, proxy=True):
        if self.parent:
            parent = self.parent.convertTo(manager)
        entity = manager.proxyCls(self, manager=manager)
        if self.parent:
            parent.addChild(entity)
        return entity
    
    # Children
    def addChild(self, entity):
        if entity not in self._children:
            if entity.absoluteName and self.findAbsoluteName(entity.absoluteName):
                raise Exception(f'An entity/property named {entity.absoluteName} already exists within {self.root.reference()}! \
                                 Names should be uniques.')
            entity.parent = self
            self._children.append(entity)
            # if entity.PHYSICAL:
            #     self.physicals.append(entity)
            if self.activated:
                entity.activate()

    def removeChild(self, entity):
        if entity in self._children:
            #del self._children[entity.name]
            if self.activated:
                entity.deactivate()
            self._children.remove(entity)
            # if entity.PHYSICAL:
            #     self.physicals.remove(entity)
            entity.parent = None

    def clearChildren(self):
        entities = list(self._children)
        for entity in entities:
            self.removeChild(entity)

    @property
    def root(self):
        if self.parent:
            return self.parent.root
        return self

    @property
    def manager(self):
        return self.root._manager

    def isRoot(self):
        return not self.parent and self.kind == 'root'
    
    def child(self, filter_=None):
        # self.findAbsoluteName(name, fromRoot=fromRoot, onlyEntities=True)
        return (self.children(filter_) + [None])[0]

    def children(self, filter_=None):
        if not filter_:
            return list(self._children)

        filtered = self._filterChild(filter_)
        return list(filter(filtered, self._children))

    def cascadingChild(self, filter_=None):
        children = self.children(filter_)
        return (children + [child for entity in children for child in entity.cascadingChildren(filter_)] + [None])[0]

    def cascadingChildren(self, filter_=None):
        children = self.children(filter_)
        return children + [child for entity in children for child in entity.cascadingChildren(filter_)]

    def _filterChild(self, filter_):
        if not filter_:
            filtered = lambda child: True
        if filter_[0] == '#':
            filtered = lambda child: child.absoluteName == filter_[1:]
        elif ':' in filter_:
            kind, index = filter_.split(':')
            filtered = lambda child: child.kind == kind and child.indexKind == index
        else:
            filtered = lambda child: child.kind == filter_
        return filtered

    # Properties and children
    def findAbsoluteName(self, name, fromRoot=True, onlyEntities=False):
        if not name:
            return None
        children = (self.root if fromRoot else self).cascadingChildren()
        for child in children:
            if child.absoluteName == name:
                return child
            if not onlyEntities:
                properties = child.cascadingProperties()
                for prop in properties:
                    if prop.absoluteName == name:
                        return prop
        return None
    
    # Properties
    @property
    def _proxifiedProperties(self):
        return self._properties

    def addProperty(self, prop):
        self._properties[prop.name] = prop

    def removeProperty(self, prop):
        if prop.name in self._properties:
            del self._properties[prop.name]

    def propertyItem(self, filter_=None):
        return (self.properties(filter_) + [None])[0]

    def properties(self, filter_=None, visual=None):
        if not filter_:
            properties = list(self._proxifiedProperties.values())
        else:
            # Omit first dot
            if filter_[0] == '.':
                filter_ = filter_[1:]
            if filter_[0] == '#':
                def filtered(property):
                    return property.absoluteName == filter_[1:]
            else:
                def filtered(property):
                    return property.name == filter_

            properties = list(filter(filtered, self._proxifiedProperties.values()))
        if visual is not None:
            properties = [prop for prop in properties if prop.visual == visual]
        return properties

    def cascadingProperties(self, filter_=None, visual=None):
        filterChildren, filterProperties = None, None
        if filter_:
            filterChildren, filterProperties = (
                filter_.split('.') + [None])[:2]
        properties = []
        if not filterChildren or self._filterChild(filterChildren)(self):
            properties = self.properties(filterProperties, visual=visual)
        return properties + \
            [property for entity in self.children(
                filterChildren) for property in entity.cascadingProperties(filter_, visual=visual)]

    def cascadingProperty(self, filter_=None):
        return (self.cascadingProperties(filter_) + [None])[0]
    
    def visualProperties(self):
        return self.properties(visual=True)

    def cascadingVisualProperties(self):
        return self.cascadingProperties(visual=True)
    
    def visualPropertyNames(self):
        properties = self.root.cascadingVisualProperties()
        return set([(prop.name, prop.dim) for prop in properties])

    def observeVisualProperties(self):
        values = []
        for name, dim in self.visualPropertyNames():
            prop = self.propertyItem(f'.{name}')
            if prop:
                values += prop.observePlain().tolist()
            else:
                values += [0] * dim
        return values
    
    def observe(self, spaces=None, formatParameters=None):
        spaces = spaces.convertTo(
            self.manager, kind=SpaceKind.BASIC) if spaces else None
        if self.filterObservables:
            obsSpaces = [x.space for x in self.filterObservables]
            if not spaces:
                spaces = obsSpaces
            else:
                spaces = [x for x in spaces if x in obsSpaces]

        ys = []
        # spaces = spaces if spaces else self.outcomeSpaces
        for property_ in self.cascadingProperties():
            if property_.observable() and property_.bounded:
                y = property_.observe(formatParameters=formatParameters)
                if spaces is None or y.space in spaces:
                    ys.append(y)
        return Observation(*ys)

    def observeFrom(self, spaces=None, formatParameters=None):
        return self.world.observe(spaces=spaces, formatParameters=formatParameters)

    def currentContext(self, dataset=None, spaces=None):
        obs = self.observe(spaces=spaces)
        if dataset:
            obs = obs.convertTo(dataset)
        return obs
    
    # Routines
    def activate(self):
        if self.activated:
            return
        self.activated = True

        self._activate()
        for e in self._children:
            e.activate()

        for p in self._properties.values():
            p._activate()

    def deactivate(self):
        if not self.activated:
            return
        self.activated = False

        self._deactivate()
        for e in self._children:
            e.deactivate()

    def _activate(self):
        pass

    def _deactivate(self):
        pass

    # Discretization
    @property
    def discretizeStates(self):
        return self.root._discretizeStates

    @discretizeStates.setter
    def discretizeStates(self, discretizeStates):
        if discretizeStates == self._discretizeStates:
            return
        assert(self.root.CAN_BE_DISCRETIZED)
        self._discretizeStates = discretizeStates
        self._discretizationChanged(False, discretizeStates)

    @property
    def discretizeActions(self):
        return self.root._discretizeActions

    @discretizeActions.setter
    def discretizeActions(self, discretizeActions):
        if discretizeActions == self._discretizeActions:
            return
        assert(self.root.CAN_BE_DISCRETIZED)
        self._discretizeActions = discretizeActions
        self._discretizationChanged(True, discretizeActions)

    def _discretizationChanged(self, action, discrete):
        pass

    @property
    def discreteStates(self):
        return self.discretizeStates or self.root.DISCRETE_STATES

    @property
    def discreteActions(self):
        return self.discretizeActions or self.root.DISCRETE_ACTIONS

    @property
    def world(self):
        # Supposed to be the root
        return self.root

    @property
    def engine(self):
        return self.manager.engine

    def observables(self):
        return [p for p in self.cascadingProperties() if p.observable()]

    def actions(self):
        actions = [p for p in self.cascadingProperties() if p.controllable()]
        if self.discretizeActions:
            actions = {(p, name): p.space.point(params)
                       for p in actions for name, params in p.actions.items()}
        return Ensemble(actions)
    
    # String
    def fullname(self):
        return self.parent.fullname() + "." + self.absoluteName if self.parent else ""

    def reference(self, short=False):
        if short:
            if self.absoluteName:
                s = f'#{self.absoluteName}'
            else:
                s = f'{self.kind}:{self.indexKind}'
        else:
            s = f'{self.kind}:{self.indexKind}'
            if self.absoluteName:
                s += f'#{self.absoluteName}'
        return s

    def __repr__(self):
        s = f'{self.kind}:{self.indexKind}'
        if self.disconnected:
            s = colorText('❌', Colors.RED) + s
        if self.absoluteName:
            s += f'#{self.absoluteName}'
        if self.parent:
            s += f' bound to {self.parent.reference()}'
        return s
    

class ProxyEntity(Entity):
    def __init__(self, entity, manager=None):
        manager = parameter(manager, entity.manager)
        super().__init__(entity.kind, entity.absoluteName,
                         manager=manager)

        self.proxy = entity
        self.native = entity.native
    
    @property
    def _proxifiedProperties(self):
        _properties = dict(self.proxy._properties)
        _properties.update(self._properties)
        return _properties

