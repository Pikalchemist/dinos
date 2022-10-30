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

from direct.actor import Actor
from panda3d.core import Point2, Point3, Vec2, Vec3, Vec4, Material
from dinos.environments.engines.tools.panda import PandaTools

from dinos.representation.physical_entity import PhysicalEntity
from dinos.representation.property import MethodObservable, AttributeObservable, MethodEffector
from .cylinder import Cylinder


class Agent(Cylinder):
    def __init__(self, coords=(480, 300), radius=20, name='', omni=True, onlyX=False, xydiscretization=20):
        super().__init__(coords, radius, movable=True, name=name,
                         color=(0, 0, 0), xydiscretization=xydiscretization)
        self.markAsHost()

        self.onlyX = onlyX
        self.omni = omni

        self.mass = 2000.

        # MethodObservable(self, "position", self.absolutePosition)
        MethodObservable(self, "lidar", self.lidar,
                         discretization=50).learnable = False
        # AttributeObservable(self, "color", 'color')
        # AttributeObservable(self, "radius", 'radius')
        # AttributeObservable(self, "mass", 'mass')
        actions = {
            'down': (0., 1.0),
            'up': (0., -1.0),
            'left': (-1.0, 0.),
            'right': (1.0, 0.),
        }
        MethodEffector(self, "move", self.move,
                       dim=1 if self.onlyX else 2, actions=actions)
        # MethodEffector(self, "toggle", self.toggle, dim=0)

        self.sufaceWidth = radius * 2 + 10
        self.surface = pygame.Surface(
            (self.sufaceWidth, self.sufaceWidth), pygame.SRCALPHA)

    def lidar(self, property=None):
        return self.lidarFromPos(self.shape, self.engine.physics, self.shape.body.position, self.onlyX)

    @staticmethod
    def lidarFromPos(ownShape, physics, position, onlyX):
        # Distances[0 E, 1 SE, 2 S, 3 SW, 4 W, 5 NW, 6 N, 7 NE]
        number = 2 if onlyX else 8
        maxdist = 70
        width = 10

        dirs = np.linspace(0, 2*np.pi, number + 1)[:-1]
        distances = np.zeros(number)
        for i, dir in enumerate(dirs):
            d = Vec2d(maxdist, 0.0).rotated(dir)
            hits = physics.segment_query(position, position + d,
                                                width, pymunk.ShapeFilter())
            dists = [info.alpha *
                     maxdist for info in hits if info.shape != ownShape]
            distances[i] = min(dists) if dists else maxdist
        return distances

    def toggle(self, parameters, property=None):
        self.shape.body.position = (
            self.shape.body.position.x + 20, self.shape.body.position.y)

    def move(self, parameters, property=None):
        # print(f'MOVE! {parameters}')
        self.performAction(self._move, args=(
            parameters, property), duration=0.6, step=0.05)

    def _move(self, parameters, property=None):
        # print("MOVE")
        # print(self.absolutePosition())
        # print(parameters)
        wl = min(max(parameters[0], -1), 1)  # ** 3
        wr = 0. if self.onlyX else min(max(parameters[1], -1), 1)  # ** 3
        noise = 0.
        pw = 10000

        # self.body.apply_impulse_at_local_point(Vec2d(1.0, 0.0) * pw * float(wl), (0, 0))

        if self.omni:
            self.body.apply_impulse_at_local_point(Vec2d(1.0, 0.0) * pw * float(wl + random.uniform(-noise, noise)) +
                                                   Vec2d(0.0, 1.0) * pw * float(wr + random.uniform(-noise, noise)),
                                                   (0, 0))
            if wl != 0. or wr != 0.:
                self.direction = np.arctan2(wl, wr)
        else:
            pass

    def draw(self, base, drawOptions):
        if drawOptions.pygame:
            pygame.draw.circle(base, (224, 224, 0), list(
                map(int, self.absolutePosition())), self.radius, 1)

            pygame.draw.circle(self.surface, (224, 224, 224),
                            (self.sufaceWidth // 2, self.sufaceWidth // 2), self.radius)
            pygame.draw.circle(self.surface, self.color, (self.sufaceWidth //
                                                        2, self.sufaceWidth // 2), self.radius, 1)

            # Orientation
            w = 10
            points = [(self.sufaceWidth // 2 + self.radius, self.sufaceWidth // 2),
                    (self.sufaceWidth // 2 + self.radius -
                    w, self.sufaceWidth // 2 - w // 2),
                    (self.sufaceWidth // 2 + self.radius - w, self.sufaceWidth // 2 + w // 2)]
            pygame.draw.polygon(self.surface, self.color, points)

            # Wheels
            for pos in (-1, 1):
                pygame.draw.line(self.surface, self.color,
                                (self.sufaceWidth // 2 - self.radius // 2,
                                self.sufaceWidth // 2 + pos * self.radius // 2),
                                (self.sufaceWidth // 2 + self.radius // 2, self.sufaceWidth // 2 + pos * self.radius // 2), 3)

            surface = pygame.transform.rotate(
                self.surface, np.rad2deg(self.direction) - 90.)
            base.blit(surface, list(
                map(lambda x: int(x) - surface.get_width() // 2, self.absolutePosition())))
        if drawOptions.panda:
            if not self.pandaInit:
                self.pandaInit = True

                self.robotNode = base.render.attachNewNode('robot axis')
                # self.robotModel = Actor.Actor('panda-model', {'walk': 'panda-walk4'})
                self.robotModel = base.loader.loadModel('robot.bam')
                # self.robotModel = base.loader.loadModel('robot.egg')
                self.robotModel.reparentTo(self.robotNode)
                self.robotModel.setScale(self.radius / 0.7)
                # self.robotModel.setScale(0.5)
                # self.pandaWalk = self.robotModel.actorInterval('walk', playRate=1.8)
                # self.pandaWalk.loop()

                # for s in ('Base'):#, 'Lidar', 'Wheel'):
                #     material = self.robotModel.findMaterial(s)
                #     # material = Material()
                #     # material = m
                #     material.setShininess(5.)
                #     material.setBaseColor((0, 0, 1, 1))
                #     # material.setDiffuse((0, 0, 1, 1))
                #     # material.setAmbient((0, 1, 0, 1))
                #     # material.setEmission((0.5, 0.5, 0.5, 1))
                #     # material.setSpecular((1, 1, 1, 1))
                #     print(material)
                #     # print(m)
                #     # self.robotModel.replaceMaterial(m, material)

                # material = self.robotModel.findMaterial('Lidar')
                # material.setBaseColor((1, 1, 1, 1))

                self.robotModel.setColor(0.4, 0.4, 0.45, 1)

                # print(self.robotModel.findAllMaterials())
                # self.robotNode.setShaderAuto()
            self.robotNode.setPos(*self.body.position, 10.)
            self.robotNode.setHpr(-np.rad2deg(self.direction) - 90, 0., 0.)

