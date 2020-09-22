import numpy as np
from pymunk import Vec2d

from dino.environments.scene import SceneSetup
from dino.evaluation.tests import UniformGridTest

from .environment import PlaygroundEnvironment
from .cylinder import Cylinder
from .agent import Agent
from .wall import Wall


class EmptyRoomScene(SceneSetup):
    DESCRIPTION = 'Just a mobile robot with no obstacle and no other objects.\n' +\
                  'Use: learning how to move.'
    RESET_ITERATIONS = 5

    def _setup(self):
        self.iterationReset = 0

        # self.cylinder = Cylinder((200, 400))
        self.world.addChild(Agent((300, 300), name='Agent'))
        # self.world.addChild(Cylinder((200, 300), name='Cylinder1'))
        # self.world.addChild(Cylinder((500, 300), name='Cylinder2'))
        # Add agent
        # self.agent = Agent((200, 400), radius=30, name='agent',
        #                    omni=True, xydiscretization=self.world.xydiscretization)
        # self.world.addEntity(self.agent)

        # Walls
        w = 5
        self.world.addChild(Wall((270.0, 200.0), (270.0, 400.0), w))
        # self.walls = [Wall((50.0, 50.0), (550.0, 50.0), w),
        #               Wall((550.0, 50.0), (550.0, 500.0), w),
        #               Wall((550.0, 500.0), (50.0, 500.0), w),
        #               Wall((50.0, 500.0), (50.0, 50.0), w),
        #               Wall((285.0, 380.0), (315.0, 380.0), w),
        #               Wall((285.0, 410.0), (315.0, 410.0), w)]
        # self.walls = self.walls[:-2]
        # for wall in self.walls:
        #     self.world.addChild(wall)

    def _setupTests(self):
        boundaries = [(100, 500), (100, 500)]
        self.addTest(UniformGridTest(self.world.cascadingProperty('Agent.position').space, boundaries, numberByAxis=2))
    
    def setupEpisode(self, config):
        self.iterationReset += 1
        if self.iterationReset >= self.RESET_ITERATIONS:
            self.iterationReset = 0
            self.world.child('Agent').body.position = (300, 300)
        # pos = self.world.child('Agent').body.position
        # self.world.child('Cylinder').body.position = pos + Vec2d(40. + np.random.uniform(0.), 0).rotated(np.random.uniform(2*np.pi))

    def setupIteration(self, config):
        pass

    def setupPreTest(self, test):
        self.reset()

    def _reset(self):
        pass
        # for obj in [self.agent]:
        #     obj.body.position = obj.coordsInit


PlaygroundEnvironment.registerScene(EmptyRoomScene, True)
