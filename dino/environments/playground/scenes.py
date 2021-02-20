import numpy as np
from pymunk import Vec2d

import random

from dino.environments.scene import SceneSetup
from dino.evaluation.tests import UniformGridTest, PointsTest

from .environment import PlaygroundEnvironment
from .cylinder import Cylinder
from .button import Button
from .agent import Agent
from .wall import Wall


class BaseScene(SceneSetup):
    RESET_ITERATIONS = 10

    # Setup
    def _baseSetup(self):
        self.iterationReset = 0

    # Resets
    def countReset(self, forceReset=False):
        self.iterationReset += 1
        if self.iterationReset >= self.RESET_ITERATIONS or forceReset:
            self.iterationReset = 0
            return True
        return False

    def setupIteration(self, config):
        pass

    def setupPreTest(self, test):
        self.reset()

    def _reset(self):
        pass
        # for obj in [self.agent]:
        #     obj.body.position = obj.coordsInit


class EmptyScene(BaseScene):
    DESCRIPTION = 'Just a mobile robot with no obstacle and no other objects.\n' +\
                  'Use: learning how to move.'

    # Setup
    def _setup(self):
        self._baseSetup()

        # Add agent
        self._setupAgent()

        # Add cylinders
        # self.world.addChild(Cylinder((400, 500), name='Cylinder3', color=(240, 0, 0), movable=False))
        # self.world.addChild(Cylinder((300, 500), name='Cylinder4', color=(240, 0, 0), movable=False))

        # self.world.addChild(Button((50, 50), name='Button1'))
        # self.world.addChild(Button((500, 500), name='Button2'))

        # self.world.addChild(Cylinder((200, 300), name='Cylinder1'))
        # self.world.addChild(Cylinder((500, 300), name='Cylinder2', color=(128, 224, 0)))
    

        # self.agent = Agent((200, 400), radius=30, name='agent',
        #                    omni=True, xydiscretization=self.world.xydiscretization)
        # self.world.addEntity(self.agent)
    
    def _setupAgent(self):
        self.world.addChild(Agent((300, 300), name='Agent'))

    # Tests
    def _setupTests(self):
        self._testAgentMoving()

    def _testAgentMoving(self):
        points = [(200, 200),
                  (200, 400),
                  (400, 400),
                  (400, 200)]
        test = PointsTest('agent-moving', self.world.cascadingProperty('Agent.position').space, points, relative=False)
        self.addTest(test)

    # Resets
    def setupEpisode(self, config, forceReset=False):
        if self.countReset(forceReset):
            self._resetAgent()
    
    def setupPreTest(self, test):
        self.world.child('Agent').body.position = (350, 300)
    
    def setupPreTestPoint(self, test, point):
        self.world.child('Agent').body.position = (350, 300)
    
    def _resetAgent(self, rand=True):
        self.world.child('Agent').body.position = (random.choice([100, 200, 300, 400]), random.randint(150, 450)) if rand else (300, 300)


class RoomWithWallsScene(EmptyScene):
    DESCRIPTION = 'Just a mobile robot with walls and no other objects.\n' +\
                  'Use: learning how to move.'

    # Setup
    def _setup(self):
        self._baseSetup()

        self._setupAgent()
        self._setupWalls()

    def _setupWalls(self):
        self.walls = []

        # outer walls
        outw = 6
        outw2 = outw // 2
        self.walls += [Wall((50.0, 50.0), (550.0, 50.0), outw),
                       Wall((550.0, 50.0 - outw2 + 1), (550.0, 500.0 + outw2), outw),
                       Wall((550.0, 500.0), (50.0, 500.0), outw),
                       Wall((50.0, 500.0 + outw2), (50.0, 50.0 - outw2 + 1), outw)]

        # inner walls
        w = 20
        self.walls += [Wall((260.0, 200.0), (260.0, 400.0), w),
                       Wall((335.0, 50.0), (335.0, 290.0), w)]
        # self.walls += [Wall((285.0, 380.0), (315.0, 380.0), w),
        #                Wall((285.0, 410.0), (315.0, 410.0), w)]

        for wall in self.walls:
            self.world.addChild(wall)

    # Tests
    def _testAgentMoving(self):
        points = [(300, 200),
                  (300, 400),
                  (400, 300)]
        test = PointsTest('agent-moving', self.world.cascadingProperty('Agent.position').space, points, relative=False)
        self.addTest(test)

        points = [(150, 250),
                  (350, 250),
                  (150, 350),
                  (350, 350)]
        test = PointsTest('agent-moving-obstacles', self.world.cascadingProperty('Agent.position').space, points, relative=False)

        def prePoint(point):
            point = point.plain()
            self.world.child('Agent').body.position = (350 if point[0] < 260 else 150, 250 if point[1] < 300 else 350)

        test.prePoint = prePoint
        self.addTest(test)


class OneCylinderScene(EmptyScene):
    DESCRIPTION = 'Just a mobile robot with no obstacle and no other objects.\n' +\
                  'Use: learning how to move.'

    # Setup
    def _setup(self):
        self._baseSetup()

        self._setupAgent()

        # Add cylinders
        self._setupCylinder1()

        # self.world.addChild(Cylinder((200, 300), name='Cylinder1'))
        # self.world.addChild(Cylinder((500, 300), name='Cylinder2'))
        # self.world.addChild(Cylinder((400, 500), name='Cylinder3', color=(240, 0, 0), movable=False))
        # self.world.addChild(Cylinder((300, 500), name='Cylinder4', color=(240, 0, 0), movable=False))

        # self.world.addChild(Button((50, 50), name='Button1'))
        # self.world.addChild(Button((500, 500), name='Button2'))

        # self.world.addChild(Cylinder((200, 300), name='Cylinder1'))
        # self.world.addChild(Cylinder((500, 300), name='Cylinder2', color=(128, 224, 0)))

        # self.agent = Agent((200, 400), radius=30, name='agent',
        #                    omni=True, xydiscretization=self.world.xydiscretization)
        # self.world.addEntity(self.agent)

    def _setupCylinder1(self):
        self.world.addChild(Cylinder((200, 300), name='Cylinder1'))

    def _setupCylinder2(self):
        self.world.addChild(Cylinder((500, 300), name='Cylinder2'))

    def _setupCylinder3Fixed(self):
        self.world.addChild(Cylinder((400, 500), name='Cylinder3', color=(240, 0, 0), movable=False))

    # Tests
    def _setupTests(self):
        self._testAgentMoving()
        self._testMovingCylinder()

    def _testMovingCylinder(self):
        self.addTest(PointsTest('cylinder1-moving', self.world.cascadingProperty(
            '#Cylinder1.position').space, [[-25, 0]], relative=True))

    # Resets
    def setupEpisode(self, config, forceReset=False):
        if self.countReset(forceReset):
            self._resetAgent()
            self._resetCylinders()

    def setupPreTest(self, test):
        self._resetAgent()
        self.world.child('#Cylinder1').body.position = (200, 250)
        if self.world.child('#Cylinder2'):
            self.world.child('#Cylinder2').body.position = (300, 150)

    def _resetCylinders(self):
        pos = self.world.child('Agent').body.position

        distance = 40. if self.environment.iteration < 100 else 120.

        obj = self.world.child('#Cylinder1').body
        if self.world.child('#Cylinder2'):
            if random.uniform(0, 1) < 0.5:
                obj2 = obj
                obj = self.world.child('#Cylinder2').body
            else:
                obj2 = self.world.child('#Cylinder2').body
            obj2.position = pos + Vec2d(240. + np.random.uniform(0.), 0).rotated(np.random.uniform(2*np.pi))
        obj.position = pos + Vec2d(distance + np.random.uniform(0.), 0).rotated(np.random.uniform(2*np.pi))


class OneCylinderPlusFixedScene(OneCylinderScene):
    DESCRIPTION = 'Just a mobile robot with no obstacle and no other objects.\n' +\
                  'Use: learning how to move.'

    # Setup
    def _setup(self):
        self._baseSetup()

        self._setupAgent()

        # Add cylinders
        self._setupCylinder1()
        self._setupCylinder3Fixed()


PlaygroundEnvironment.registerScene(EmptyScene, True)
PlaygroundEnvironment.registerScene(RoomWithWallsScene)
PlaygroundEnvironment.registerScene(OneCylinderScene)
PlaygroundEnvironment.registerScene(OneCylinderPlusFixedScene)

