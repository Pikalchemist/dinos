import pygame
import pymunk
from pymunk import Vec2d

from PIL import Image

from .entity import ProxyEntity
from .entity_manager import EntityManager
from .property import FunctionObservable


class MapProxyEntity(ProxyEntity):
    # def __init__(self, entity, manager=None):
    #     super().__init__(entity, manager=manager)

    # Handlers
    def addPropertyHandler(self, name, handler):
        property_ = self.propertyItem(name)
        # self.handlers[property_] = handler
        self.manager.addPropertyHandler(property_, handler)


class FeatureMap(EntityManager):
    def __init__(self, proxyCls=ProxyEntity):
        EntityManager.__init__(self, 'featureMap', proxyCls=proxyCls)

        self.handlers = {}
        self.generators = []

    def addPropertyHandler(self, property_, handler):
        if property_:
            self.handlers[property_] = handler
    
    def addPropertyGenerator(self, entityFilter, name, function, **kwargs):
        self.generators.append((entityFilter, name, function, kwargs))
    
    def customEntityConversion(self, entity):
        for entityFilter, name, function, kwargs in self.generators:
            if entity in self.world.cascadingChildren(entityFilter):
                FunctionObservable(entity, name, function, **kwargs, proxySpace=True)
        return entity

    def populateFrom(self, manager):
        for entity in manager.entities():
            entity.convertTo(self)
        self.world.activate()

    def update(self, values):
        pass

    def image(self):
        pass
