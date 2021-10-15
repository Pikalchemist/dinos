from .pymunk import PyMunkFeatureMap
from dinos.environments.playground import Agent


class PlaygroundFeatureMap(PyMunkFeatureMap):
    def __init__(self):
        super().__init__()
        self.addPropertyGenerator('Agent', 'lidar', self.lidar)

    def lidar(self, entity, property=None):
        return Agent.lidarFromPos(entity.shape, self.physics, entity.shape.body.position, entity.native.onlyX)
