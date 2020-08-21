from dino.environments.scene import SceneSetup
from .environment import PlaygroundEnvironment
from .cylinder import Cylinder


class EmptyRoomScene(SceneSetup):
    DESCRIPTION = 'Just a mobile robot with no obstacle and no other objects.\n' +\
                  'Use: learning how to move.'

    def _setup(self):
        self.cylinder = Cylinder((200, 400))
        self.world.addChild(self.cylinder)
        # Add agent
        # self.agent = Agent((200, 400), radius=30, name='agent',
        #                    omni=True, xydiscretization=self.world.xydiscretization)
        # self.world.addEntity(self.agent)

    # def _setupTests(self):
    #     boundaries = [(100, 500), (100, 500)]
    #     self.addTest(UniformGridTest(self.world.cascadingProperty(
    #         'Agent.position').space, boundaries, numberByAxis=2))

    def setupIteration(self):
        pass

    def setupPreTest(self, test=None):
        self.reset()

    def _reset(self):
        pass
        # for obj in [self.agent]:
        #     obj.body.position = obj.coordsInit


PlaygroundEnvironment.registerScene(EmptyRoomScene, True)
