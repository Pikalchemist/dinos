# import numpy as np
# import random
# import pygame
# from pygame.locals import *
# from pygame.color import *

from dino.environments.engines.internals.pymunk import PymunkEngine
from dino.environments.environment import Environment

# from worlds.pandaworld import PandaWorld
# from ..scene import SceneSetup, Challenge
# from ..testbench import *
# from ...utils.objects import MoveConfig

# from .objects import *


class PlaygroundEnvironment(Environment):
    NAME = 'Playground'
    DESCRIPTION = 'A simple room with a wheeled robot and some obstacles'
    ENGINE = PymunkEngine

    # def __init__(self, scene=None, options={}):
    #     super().__init__(scene, options)

    # def registerScenes(self):
    #     self.addScene(ObstructedRoomScene(), default=True)

    #     self.addScene(EmptyRoomScene())
    #     self.addScene(EmptyRoomOmniScene())

    #     self.addScene(OneObjectScene())
    #     self.addScene(MultiObjectsScene())

    #     self.addScene(Obstructed1DRoomScene())
    #     self.addScene(SpotRoomScene())
