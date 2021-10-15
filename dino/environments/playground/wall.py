import math
import random
import numpy as np

import pygame
from pygame.locals import *
from pygame.color import *
import pymunk
from pymunk import Vec2d

from panda3d.core import Point2, Point3, Vec2, Vec3, Vec4
from dino.environments.engines.tools.panda import PandaTools

from dino.representation.physical_entity import PhysicalEntity
from dino.representation.property import MethodObservable, AttributeObservable



class Wall(PhysicalEntity):
    def __init__(self, coordsFrom, coordsTo, width=40):
        super().__init__(self.__class__.__name__)
        self.width = width
        self.coordsFrom = coordsFrom
        self.coordsTo = coordsTo

        self.depth = 50
        self.pandaInit = False

    def initPhysics(self, physics):
        self.shape = pymunk.Segment(
            physics.static_body, self.coordsFrom, self.coordsTo, self.width // 2)
        self.shape.elasticity = 0.95
        self.shape.friction = 0.9
        physics.add(self.shape)

    def stopPhysics(self, physics):
        physics.remove(self.shape)

    def draw(self, base, drawOptions):
        line = self.shape
        if drawOptions.pygame:
            pygame.draw.line(base, THECOLORS["black"], (line.a.x, line.a.y), (line.b.x, line.b.y), self.width)
        if drawOptions.panda and not self.pandaInit:
            self.pandaInit = True
            a = Point2(line.a.x, line.a.y)
            b = Point2(line.b.x, line.b.y)
            print(a, b)
            v = b - a
            direction = direction = np.arctan2(v.y, v.x)# / np.math.pi * 180.

            w2 = self.width / 2
            pt = PandaTools()
            pt.color = Vec4(0.5, 0.5, 0.5, 1.)
            pt.quadri(Point3(a.x + np.cos(direction + np.math.pi / 2) * w2, a.y + np.sin(direction + np.math.pi / 2) * w2, self.depth),
                      Point3(a.x + np.cos(direction - np.math.pi / 2) * w2, a.y + np.sin(direction - np.math.pi / 2) * w2, self.depth),
                      Point3(b.x + np.cos(direction - np.math.pi / 2) * w2, b.y + np.sin(direction - np.math.pi / 2) * w2, self.depth),
                      Point3(b.x + np.cos(direction + np.math.pi / 2) * w2, b.y + np.sin(direction + np.math.pi / 2) * w2, self.depth), 0)
            node = pt.end()

            nodePath = base.render.attachNewNode(node)
            nodePath.setPos(0., 0., 0.)
            nodePath.setShaderAuto()
