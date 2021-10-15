import math
import random
import numpy as np
from scipy.spatial.distance import euclidean

import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
from pymunk import Vec2d
import pymunk.pygame_util

from dinos.representation.live_entity import LiveEntity
from dinos.representation.property import MethodObservable, AttributeObservable


class Button(LiveEntity):
    def __init__(self, coords, radius=20, name='', xydiscretization=20):
        super().__init__(self.__class__.__name__, name)
        self.coordsInit = coords
        self.coords = Vec2d(coords)

        self.radius = radius

        MethodObservable(self, 'pressed', self.pressed,
                         discretization=1)

        # Visual properties have to be the same for all entities!
        # AttributeObservable(self, 'pressed', 'pressed',
        #                     discretization=1, visual=True)
        # AttributeObservable(self, 'radius', 'radius', discretization=10)
        # AttributeObservable(self, 'mass', 'mass', discretization=100)

    # def relativePosition(self, property=None):
    #     # - self.env.agents[0].shape.body.position
    #     relativePosition = self.body.position
    #     return [relativePosition.x, relativePosition.y]

    def pressed(self, property=None):
        for cylinder in self.world.cascadingChildren('Cylinder'):
            if self.coords.get_distance(Vec2d(cylinder.absolutePosition())) < cylinder.radius + self.radius:
                return True
        return False
        # return [self.shape.body.position.x, self.shape.body.position.y]

    def draw(self, base, drawOptions):
        if drawOptions.pygame:
            pygame.draw.circle(base, (0, 0, 200), (int(self.coords.x), int(self.coords.y)), self.radius)
