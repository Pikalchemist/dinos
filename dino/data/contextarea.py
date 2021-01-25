import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

from exlab.interface.graph import Graph
from exlab.interface.serializer import Serializable
from exlab.utils.io import parameter

from . import operations


class ContextSpatialization(Serializable):
    MAX_AREAS = 100
    NN_NUMBER = 10
    MINIMUM_POINTS = 100
    THRESHOLD_ADD = 0.05
    THRESHOLD_DEL = -0.02
    THRESHOLD_RESET = 0.05

    def __init__(self, model, space, boolean=False):
        self.model = model
        self.space = space
        self.evaluatedSpace = self.model.contextSpace
        self.boolean = boolean
        self.resetAreas()

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['areas', 'stability'])
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            raise Exception('No full deserializer is available for this class')
        return super()._deserialize(dict_, serializer, obj)

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        serializer = serializer.clone(values={'.area.manager': self})

        self.stability = dict_.get('stability', 0)
        for areaDict in dict_.get('areas', []):
            area = serializer.deserialize(areaDict)
            self._addArea(area)
    
    @property
    def dataset(self):
        return self.model.dataset
    
    # def continueFrom(self, cs):
    #     for area in cs.areas:
    #         self._addArea(area.cloneTo(self))

    def _addArea(self, area):
        if len(self.areas) >= self.MAX_AREAS:
            return

        self.centers[len(self.areas), :] = area.center.npPlain()
        self.areas.append(area)
        self.findAllPointAreas()
        self.model.invalidateCompetences()

    def _removeArea(self, area):
        index = self.areas.index(area)
        self.centers[index:-1, :] = self.centers[index+1:, :]
        self.areas.remove(area)
        self.findAllPointAreas()
        self.model.invalidateCompetences()
    
    def resetAreas(self):
        self.areas = []
        self.centers = np.zeros((self.MAX_AREAS, self.space.dim))
        self.stability = 0
        self.model.invalidateCompetences()
    
    def allTrue(self):
        self.resetAreas()
        self._addArea(ContextArea(self, self.space.zero(), np.full(self.evaluatedSpace.dim, True)))

    def findArea(self, point, space=None):
        if not self.areas:
            return None
        point = point.projection(parameter(space, self.space))
        nearest, _ = operations.nearestNeighborsFromDataContiguous(self.centers[:len(self.areas)], point.npPlain())
        return self.areas[nearest[0]]
    
    def columns(self, goal, space=None):
        area = self.findArea(goal, space)
        # print(area)
        if not area:
            return np.full(self.evaluatedSpace.dim, False)
        return area.columns
    
    def findAllPointAreas(self):
        if not self.areas:
            return

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

        # print('---')
        # print(c)
        # print(point)
        bestAdd = ()
        bestDel = ()
        # print(point)

        area = self.findArea(point)
        if area:
            stability = area.stability
            area.addPoint(id_)
            area.stability += 1
        else:
            stability = self.stability
            self.stability += 1

        probality = max(0.05, np.exp(-stability * 0.1))
        # print(self.model, stability, probality)

        fullCompAllFalse = None
        fullCompAllTrue = None
        if area and random.uniform(0, 1) < probality * 0.1:
            fullComp = self.model.competence(precise=True)

            columnsFalse = np.full(self.evaluatedSpace.dim, False)
            fullCompAllFalse = self.model.competence(precise=True, contextColumns=columnsFalse)

            columnsTrue = np.full(self.evaluatedSpace.dim, True)
            fullCompAllTrue = self.model.competence(precise=True, contextColumns=columnsTrue)

            bestFullComp = max(fullCompAllFalse, fullCompAllTrue)
            columns = columnsTrue if fullCompAllTrue > fullCompAllFalse else columnsFalse

            if fullComp < bestFullComp - self.THRESHOLD_RESET * 2:
                # print('Reset all areas!')
                self.resetAreas()
                area = ContextArea(self, point, columns)
                self._addArea(area)

        if random.uniform(0, 1) < probality:
            currentColumns = self.columns(point)
            c = self.model.competence(onlyIds=nearest, contextColumns=currentColumns)

            dims = np.arange(self.evaluatedSpace.dim)
            np.random.shuffle(dims)
            n = random.randint(1, min(1 + int(self.evaluatedSpace.dim * 0.4), 3))
            dims = dims[:n]
            # dims = range(self.evaluatedSpace.dim)  # dims[:n]

            # print(dims)
            # for i in range(self.evaluatedSpace.dim):
            for i in dims:
                columns = np.copy(currentColumns)
                columns[i] = not columns[i]
                # print(columns)
                nc = self.model.competence(onlyIds=nearest, contextColumns=columns)
                p = nc - c
                if columns[i] and p >= self.THRESHOLD_ADD and (not bestAdd or p > bestAdd[0]):
                    bestAdd = (p, i, columns)
                if not columns[i] and p >= -self.THRESHOLD_DEL and (not bestDel or p > bestDel[0]):
                    bestDel = (p, i, columns)

            if (bestAdd or bestDel) and not fullCompAllFalse:
                columnsFalse = np.full(self.evaluatedSpace.dim, False)
                fullCompAllFalse = self.model.competence(
                    precise=True, contextColumns=columnsFalse)

                columnsTrue = np.full(self.evaluatedSpace.dim, True)
                fullCompAllTrue = self.model.competence(
                    precise=True, contextColumns=columnsTrue)
                
                bestFullComp = max(fullCompAllFalse, fullCompAllTrue)

            for best, deletion, verb in ((bestAdd, False, 'add'), (bestDel, False, 'del')):
                if best:
                    p, i, newColumns = best
                    # print(f'Should {verb} context column {i} (+{p}) around {point}')
                    createNew = False

                    if area:
                        if area.attempt(newColumns, deletion):
                            # print(f'Updating current area')
                            previousColumns = area.columns
                            area.columns = newColumns
                        else:
                            # print(f'Conflict trying to update current area, creating a new one')
                            createNew = True
                    else:
                        # print(f'No existing area! Adding one')
                        createNew = True
                    if createNew:
                        newArea = ContextArea(self, point, newColumns)
                        self._addArea(newArea)
                    fullComp = self.model.competence(precise=True)
                    if fullComp < bestFullComp - self.THRESHOLD_RESET:
                        # print('Aborting')
                        if createNew:
                            self._removeArea(newArea)
                        else:
                            area.columns = previousColumns
                    elif not createNew:
                        area.stability = 0
            # self.model.restore()
    
    # Visual
    def visualizeAreas(self, options={}):
        g = Graph(title=f'Context areas from {self.space}', options=options)
        areas = {}
        for area in self.areas:
            columns = repr(area.columns.tolist())
            if columns in areas.keys():
                areas[columns] = np.vstack((areas[columns], self.space.getData(area.ids)))
            else:
                areas[columns] = self.space.getData(area.ids)
        for columns, data in areas.items():
            g.scatter(data, label=columns)
        return g


class ContextArea(Serializable):
    def __init__(self, manager, center, columns):
        self.manager = manager
        self.center = center
        self.columns = columns
        self.ids = np.array([])
        self.stability = 0
    
    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['center', 'columns', 'stability'])
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls(serializer.get('.area.manager'),
                      serializer.deserialize(dict_.get('center')),
                      np.array(serializer.deserialize(dict_.get('columns'))))
        return super()._deserialize(dict_, serializer, obj)

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        self.stability = dict_.get('stability', 0)
    
    # def cloneTo(self, manager):
    #     columns = np.full(manager.evaluatedSpace.dim, False)

    #     posn = 0
    #     poso = 0
    #     for sn in manager.evaluatedSpace.cols:
    #         for so in self.manager.evaluatedSpace.cols:
    #             if sn.matches(so):
    #                 columns[posn:posn + sn.dim] = self.columns[poso:poso + so.dim]
    #                 break
    #             poso += so.dim
    #         posn += sn.dim

    #     new = self.__class__(manager, self.center, columns)
    #     return new

    def addPoint(self, id_):
        self.ids = np.append(self.ids, id_)
    
    def attempt(self, columns, deletion):
        c = self.manager.model.competence(
            onlyIds=self.ids, contextColumns=self.columns)
        nc = self.manager.model.competence(
            onlyIds=self.ids, contextColumns=columns)
        if deletion:
            return nc - c > -ContextSpatialization.THRESHOLD_DEL
        else:
            return nc - c > ContextSpatialization.THRESHOLD_ADD


    def __repr__(self):
        if self not in self.manager.areas:
            return f'Area centered on {self.center}'
        return f'Area {self.manager.areas.index(self)} centered on {self.center}'
