import numpy as np

from . import operations


class ContextSpatialization(object):
    MAX_AREAS = 1000
    NN_NUMBER = 20
    MINIMUM_POINTS = 500
    THRESHOLD_ADD = 0.05
    THRESHOLD_DEL = 0.001

    def __init__(self, model, space):
        self.model = model
        self.space = space
        self.areas = []
        self.centers = np.zeros((self.MAX_AREAS, space.dim))
    
    @property
    def dataset(self):
        return self.model.dataset

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
    
    def columns(self, goal):
        area = self.findArea(goal)
        if not area:
            return np.full(self.model.contextSpace.dim, False)
        return None

    def addPoint(self, point):
        if self.space.number < self.MINIMUM_POINTS:
            return
        nearest, _ = self.space.nearest(point, n=self.NN_NUMBER)
        # self.model.saveRestrictionIds(nearest)
        c = self.model.competence(onlyIds=nearest)

        currentColumns = self.columns(point)
        # print('---')
        # print(c)
        # print(point)
        bestAdd = ()
        bestDel = ()
        for i in range(self.model.contextSpace.dim):
            columns = np.copy(currentColumns)
            columns[i] = not columns[i]
            nc = self.model.competence(onlyIds=nearest, contextColumns=columns)
            p = nc - c
            if p > 0:
                print(p)
                print(columns)
            if columns[i] and p >= self.THRESHOLD_ADD and (not bestAdd or p > bestAdd[1]):
                bestAdd = (i, p)
            if not columns[i] and p <= -self.THRESHOLD_DEL and (not bestDel or p < bestDel[1]):
                bestDel = (i, p)

        area = self.findArea(point)
        if bestAdd:
            i, p = bestAdd
            print(f'Should add context column {i} (+{p}) around {point}')

        if bestDel:
            i, p = bestDel
            print(f'Should delete context column {i} (+{p}) around {point}')
        # self.model.restore()


class ContextArea(object):
    def __init__(self, manager, center, columns):
        self.manager = manager
        self.center = center
        self.columns = columns  # self.manager.contextSpace.dim

    def __repr__(self):
        return f'Area {self.manager.areas.index(self)} centered on {self.center}'
