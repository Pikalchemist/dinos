# import numpy as np
# import random
# import pygame
# from pygame.locals import *
# from pygame.color import *

from dinos.environments.engines.internals.pymunk import PymunkEngine
from dinos.environments.environment import Environment

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
