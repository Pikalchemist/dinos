import pygame
import pymunk
from pymunk import Vec2d

from PIL import Image

from dino.representation.map import FeatureMap, MapProxyEntity


class PyMunkMapProxyEntity(MapProxyEntity):
    def __init__(self, entity, manager=None):
        super().__init__(entity, manager=manager)
        self.copyPhysics(entity)
        self.addToMap()

        # self.handlers = {}
        self.addPropertyHandler('position', self.handlerPosition)

    def copyPhysics(self, entity):
        self.shape = None
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        if hasattr(entity, 'body') and entity.body:
            self.body.position = entity.body.position
        if hasattr(entity, 'shape') and entity.shape:
            self.shape = entity.shape.copy()
            self.shape.body = self.body

    def addToMap(self):
        if self.body and self.shape:
            self.manager.physics.add(self.body, self.shape)

    def deleteFromMap(self):
        if self.body and self.shape:
            self.manager.physics.remove(self.body, self.shape)

    # Handlers
    def addPropertyHandler(self, name, handler):
        property_ = self.propertyItem(name)
        # self.handlers[property_] = handler
        self.manager.addPropertyHandler(property_, handler)

    def handlerPosition(self, value):
        if self.body:
            self.body.position = value


class PyMunkFeatureMap(FeatureMap):
    def __init__(self):
        super().__init__(proxyCls=PyMunkMapProxyEntity)

        self.physics = pymunk.Space()
        self.physics.gravity = (0.0, 0.0)
        self.physics.damping = 0.1

    def image(self):
        self.physics.step(0.0001)
        surface = pygame.Surface((800, 600))
        options = pymunk.pygame_util.DrawOptions(surface)
        pymunk.pygame_util.positive_y_is_up = False
        self.physics.debug_draw(options)

        pil_string_image = pygame.image.tostring(surface, "RGBA", False)
        pil_image = Image.frombytes(
            "RGBA", surface.get_size(), pil_string_image)

        return pil_image
