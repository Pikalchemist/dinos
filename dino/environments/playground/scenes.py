from dino.environments.scene import SceneSetup
from .environment import PlaygroundEnvironment
from .cylinder import Cylinder
from .agent import Agent


class EmptyRoomScene(SceneSetup):
    DESCRIPTION = 'Just a mobile robot with no obstacle and no other objects.\n' +\
                  'Use: learning how to move.'

    def _setup(self):
        # self.cylinder = Cylinder((200, 400))
        self.world.addChild(Agent((300, 300), name='Agent'))
        self.world.addChild(Cylinder((200, 300), name='Cylinder1'))
        self.world.addChild(Cylinder((400, 300), name='Cylinder2'))
        # Add agent
        # self.agent = Agent((200, 400), radius=30, name='agent',
        #                    omni=True, xydiscretization=self.world.xydiscretization)
        # self.world.addEntity(self.agent)

    # def _setupTests(self):
    #     boundaries = [(100, 500), (100, 500)]
    #     self.addTest(UniformGridTest(self.world.cascadingProperty(
    #         'Agent.position').space, boundaries, numberByAxis=2))

    def setupIteration(self, config):
        pass

    def setupPreTest(self, test):
        self.reset()

    def _reset(self):
        pass
        # for obj in [self.agent]:
        #     obj.body.position = obj.coordsInit


PlaygroundEnvironment.registerScene(EmptyRoomScene, True)
