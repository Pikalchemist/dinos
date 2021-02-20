import math
import random
import numpy as np

import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
from pymunk import Vec2d

from dino.representation.physical_entity import PhysicalEntity
from dino.representation.property import MethodObservable, AttributeObservable



class Wall(PhysicalEntity):
    def __init__(self, coordsFrom, coordsTo, width=40):
        super().__init__(self.__class__.__name__)
        self.width = width
        self.coordsFrom = coordsFrom
        self.coordsTo = coordsTo

    def initPhysics(self, physics):
        self.shape = pymunk.Segment(
            physics.static_body, self.coordsFrom, self.coordsTo, self.width // 2)
        self.shape.elasticity = 0.95
        self.shape.friction = 0.9
        physics.add(self.shape)

    def stopPhysics(self, physics):
        physics.remove(self.shape)

    def draw(self, screen):
        line = self.shape
        pygame.draw.line(
            screen, THECOLORS["black"], (line.a.x, line.a.y), (line.b.x, line.b.y), self.width)
