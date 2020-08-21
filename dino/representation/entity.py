'''
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import numpy as np

from exlab.interface.serializer import Serializable, Serializer
from exlab.utils.io import colorText, Colors
from exlab.utils.ensemble import Ensemble

from dino.data.data import Observation
from dino.data.space import SpaceKind


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

    def __init__(self, kind, absoluteName='', disconnected=False, spaceManager=None):
        self.absoluteName = absoluteName
        self.kind = kind

        self._spaceManager = spaceManager

        self.disconnected = disconnected
        self._discretizeStates = False
        self._discretizeActions = False

        # Indexing
        self.index = Entity.number
        Entity.number += 1
        self.indexKind = Entity.indexes.get(self.kind, 0)
        Entity.indexes[self.kind] = self.indexKind + 1

        self._properties = {}
        self._children = []
        self.physicals = []
        self.actionQueue = []
        self.parent = None
        self.activated = False

        self.filterObservables = None

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['kind', 'absoluteName', 'index', 'indexKind', 'parent', '_children', '_properties'])
        return dict_
    
    # Children
    def addChild(self, entity):
        if entity not in self._children:
            if entity.absoluteName and self.findAbsoluteName(entity.absoluteName):
                raise Exception('An entity/property named {} already exists within {}! \
                                 Names should be uniques.'.format(entity.absoluteName, self.root.reference()))
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
    def spaceManager(self):
        return self.root._spaceManager
    
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
            def filtered(child):
                return True
        if filter_[0] == '#':
            def filtered(child):
                return child.absoluteName == filter_[1:]
        else:
            def filtered(child):
                return child.kind == filter_
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
    def addProperty(self, prop):
        self._properties[prop.name] = prop

    def removeProperty(self, prop):
        if prop.name in self._properties:
            del self._properties[prop.name]

    def property(self, filter_=None):
        return (self.properties(filter_) + [None])[0]

    def properties(self, filter_=None):
        if not filter_:
            return list(self._properties.values())

        # Omit first dot
        if filter_[0] == '.':
            filter_ = filter_[1:]
        if filter_[0] == '#':
            def filtered(property):
                return property.absoluteName == filter_[1:]
        else:
            def filtered(property):
                return property.name == filter_

        return list(filter(filtered, self._properties.values()))

    def cascadingProperties(self, filter_=None):
        filterChildren, filterProperties = None, None
        if filter_:
            filterChildren, filterProperties = (
                filter_.split('.') + [None])[:2]
        properties = []
        if not filterChildren or self._filterChild(filterChildren)(self):
            properties = self.properties(filterProperties)
        return properties + \
            [property for entity in self.children(
                filterChildren) for property in entity.cascadingProperties(filter_)]

    def cascadingProperty(self, filter_=None):
        return (self.cascadingProperties(filter_) + [None])[0]
    
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
    
    # String
    def fullname(self):
        return self.parent.fullname() + "." + self.name if self.parent else ""

    def reference(self, short=False):
        if short:
            if self.absoluteName:
                s = '#{}'.format(self.absoluteName)
            else:
                s = '{}:{}'.format(self.kind, self.indexKind)
        else:
            s = '{}:{}'.format(self.kind, self.indexKind)
            if self.absoluteName:
                s += '#{}'.format(self.absoluteName)
        return s

    def __repr__(self):
        s = '{}:{}'.format(self.kind, self.indexKind)
        if self.disconnected:
            s = colorText('❌', Colors.RED) + s
        if self.absoluteName:
            s += "#{}".format(self.absoluteName)
        if self.parent:
            s += " bound to {}".format(self.parent.reference())
        return s