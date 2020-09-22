import numpy as np

from . import operations


class ContextSpatialization(object):
    MAX_AREAS = 1000
    NN_NUMBER = 20

    def __init__(self, model, space):
        self.model = model
        self.space = space
        self.areas = []
        self.centers = np.zeros((self.MAX_AREAS, space.dim))

    def _addArea(self, area):
        if len(self.areas) >= self.MAX_AREAS:
            return

        self.centers[len(self.areas), :] = area.center.npPlain()
        self.areas.append(area)

    def _removeArea(self, area):
        index = self.areas.index(area)
        self.centers[index:-1, :] = self.centers[index+1:, :]
        self.areas.append(area)

    def findArea(self, point):
        if not self.areas:
            return None
        nearest, _ = operations._nearestFromData(self.centers[:len(self.areas), :], point.npPlain())
        return self.areas[nearest[0]]

    def addPoint(self, point):
        nearest = self.space.nearest(point, n=self.NN_NUMBER)



class ContextArea(object):
    def __init__(self, manager, center, columns):
        self.manager = manager
        self.center = center
        self.columns = columns  # self.manager.contextSpace.dim

    def __repr__(self):
        return f'Area {self.manager.areas.index(self)} centered on {self.center}'
