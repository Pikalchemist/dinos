import numpy as np
from sklearn.neighbors import NearestNeighbors

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
        self.findAllPointAreas()

    def _removeArea(self, area):
        index = self.areas.index(area)
        self.centers[index:-1, :] = self.centers[index+1:, :]
        self.areas.remove(area)
        self.findAllPointAreas()

    def findArea(self, point):
        if not self.areas:
            return None
        point = point.projection(self.space)
        nearest, _ = operations._nearestFromData(self.centers[:len(self.areas)], point.npPlain())
        return self.areas[nearest[0]]
    
    def columns(self, goal):
        area = self.findArea(goal)
        # print(area)
        if not area:
            return np.full(self.model.contextSpace.dim, False)
        return area.columns
    
    def findAllPointAreas(self):
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.centers[:len(self.areas)])
        data = self.space.getData()
        _, indices = nbrs.kneighbors(data)
        indices = indices.flatten()
        for i, area in enumerate(self.areas):
            area.ids = self.space.getIds()[np.argwhere(indices == i).flatten()]

    def addPoint(self, point):
        if self.space.number < self.MINIMUM_POINTS:
            return
        point = point.projection(self.space)
        nearest, _ = self.space.nearest(point, n=self.NN_NUMBER)
        id_ = nearest[0]
        # self.model.saveRestrictionIds(nearest)
        currentColumns = self.columns(point)
        c = self.model.competence(onlyIds=nearest, contextColumns=currentColumns)

        # print('---')
        # print(c)
        # print(point)
        bestAdd = ()
        bestDel = ()
        # print(point)
        for i in range(self.model.contextSpace.dim):
            columns = np.copy(currentColumns)
            columns[i] = not columns[i]
            # print(columns)
            nc = self.model.competence(onlyIds=nearest, contextColumns=columns)
            p = nc - c
            if columns[i] and p >= self.THRESHOLD_ADD and (not bestAdd or p > bestAdd[0]):
                bestAdd = (p, i, columns)
            if not columns[i] and p >= -self.THRESHOLD_DEL and (not bestDel or p > bestDel[0]):
                bestDel = (p, i, columns)

        area = self.findArea(point)
        area.addPoint(id_)

        for best, verb in ((bestAdd, 'add'), (bestDel, 'del')):
            if best:
                p, i, newColumns = best
                print(f'Should {verb} context column {i} (+{p}) around {point}')
                createNew = False

                if area:
                    if area.attempt(newColumns):
                        print(f'Updating current area')
                        area.columns = newColumns
                    else:
                        print(f'Conflict trying to update current area, creating a new one')
                        createNew = True
                else:
                    print(f'No existing area! Adding one')
                    createNew = True
                if createNew:
                    self._addArea(ContextArea(self, point, newColumns))
        # self.model.restore()


class ContextArea(object):
    def __init__(self, manager, center, columns):
        self.manager = manager
        self.center = center
        self.columns = columns  # self.manager.contextSpace.dim
        self.ids = np.array([])
    
    def addPoint(self, id_):
        self.ids = np.append(self.ids, id_)
    
    def attempt(self, columns):
        c = self.manager.model.competence(
            onlyIds=self.ids, contextColumns=self.columns)
        nc = self.manager.model.competence(
            onlyIds=self.ids, contextColumns=columns)
        return nc - c > ContextSpatialization.THRESHOLD_ADD


    def __repr__(self):
        return f'Area {self.manager.areas.index(self)} centered on {self.center}'
