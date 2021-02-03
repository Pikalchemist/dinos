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
        # self.rawHandlers = {}
        self.generators = []
        self.rawGenerators = []
    
    def __repr__(self):
        return f'{self.__class__.__name__} (adding {", ".join([f"{g[0]}.{g[1]}" for g in self.generators])})'

    def addPropertyHandler(self, property_, handler):
        if property_:
            self.handlers[property_] = handler
    
    def addPropertyGenerator(self, entityFilter, name, function, **kwargs):
        self.generators.append((entityFilter, name, function, kwargs))
    
    def customEntityConversion(self, entity):
        for entityFilter, name, function, kwargs in self.generators:
            if entity in self.world.cascadingChildren(entityFilter):
                property_ = FunctionObservable(entity, name, function, **kwargs, proxySpace=True)
                self.rawGenerators.append(property_)
        # for name, handler in self.handlers:
        #     property_ = entity.propertyItem(name)
        #     if property_:
        #         self.rawHandlers[property_] = lambda value: handler(entity, value)
        return entity

    def populateFrom(self, manager):
        for entity in manager.entities():
            entity.convertTo(self)
        self.world.activate()
    
    def setToValues(self, values):
        for value in values:
            handler = self.handlers.get(value.space.boundProperty)
            if handler:
                handler(value.plain())
    
    def updateValues(self, values, dataset):
        for property_ in self.rawGenerators:
            found = False
            space = property_.space
            newValue = property_.observe()
            if dataset:
                newValue = newValue.convertTo(dataset)
            for i, value in enumerate(values):
                if value.space.linkedTo(space):
                    values[i] = newValue
                    found = True
                    break
            if not found:
                values.append(newValue)

    def update(self, values, dataset=None):
        self.setToValues(values)
        self.updateValues(values, dataset)
        return values

    def image(self):
        pass
