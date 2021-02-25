import math
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
    CENTER_RADIUS = 0.27
    MINIMUM_POINTS = 10
    THRESHOLD_ADD = 0.01
    THRESHOLD_DEL = 0.01
    THRESHOLD_ADD_POINT = 0.002
    THRESHOLD_DEL_POINT = 0.002
    THRESHOLD_RESET = 0.05
    THRESHOLD_VALID = 0.5
    THRESHOLD_CANT_CREATE = 0.1
    MIN_DISTANCE_CENTERS = 0.05

    def __init__(self, model, space, boolean=False):
        self.model = model
        self.space = space
        self.evaluatedSpace = self.model.contextSpace
        self.dim = self.evaluatedSpace.dim
        self.boolean = boolean

        # if 

        self.stability = 0
    
        self.resetAreas(False)
    
    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['number', 'data', 'relevances'])
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            raise Exception('No full deserializer is available for this class')
        return super()._deserialize(dict_, serializer, obj)

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        serializer = serializer.clone(values={'.area.manager': self})

        self.number = serializer.deserialize(dict_.get('number', []))
        self.data = serializer.deserialize(dict_.get('data', []))
        self.relevances = serializer.deserialize(dict_.get('relevances', []))

    @property
    def dataset(self):
        return self.model.dataset
    
    # def continueFrom(self, cs):
    #     for area in cs.areas:
    #         self._addArea(area.cloneTo(self))

    def _addArea(self, column, point, relevant):
        if self.number[column] >= self.MAX_AREAS:
            return
            # self.data[column, :-1, :] = self.data[column, 1:, :]
            # self.relevances[column, :-1] = self.relevances[column, 1:]

        self._updateArea(column, point, relevant, self.number[column])
        self.number[column] += 1

        # self.findAllPointAreas()
        self.model.invalidateCompetences()
    
    def _updateArea(self, column, point, relevant, index):
        if index == 0:
            return

        self.data[column, index, :] = point.npPlain()
        self.relevances[column, index] = relevant

        self.model.invalidateCompetences()

    # def _removeArea(self, area):
    #     column = area.column
    #     index = self.areas[column].index(area)
    #     self.centers[column, index:-1, :] = self.centers[column, index+1:, :]
    #     self.areas[column].remove(area)

    #     # self.findAllPointAreas()
    #     self.model.invalidateCompetences()

    def resetAreas(self, defaultRelevance=False):
        self.number = np.ones(self.dim, dtype=np.int)
        self.data = np.zeros((self.dim, self.MAX_AREAS, self.space.dim))
        self.relevances = np.full((self.dim, self.MAX_AREAS), defaultRelevance, dtype=np.bool)

        self.model.invalidateCompetences()
    
    def allTrue(self):
        self.resetAreas(True)

    def allFalse(self):
        self.resetAreas(False)
    
    def _groundTruthLidar(self, point, column):
        if point.norm() < 0.01:
            return False
        point = point.plain()
        direction = np.arctan2(point[1], point[0]) / np.math.pi * 180.
        col = int(np.round(direction / 45)) % 8
        # colUp = int(np.ceil(direction / 45)) % 8
        # colDown = int(np.floor(direction / 45)) % 8
        if abs(col - column) <= 2 or abs(col - column) >= 6:
            return True
        # if column in [colUp, colDown]:
        #     return 1, math.inf
        else:
            return False
    
    def _findAreaForOneColumn(self, point, column, space, projection=False, hardcoded=False):

        if projection:
            point = point.projection(parameter(space, self.space))
        # HARDCODED
        # return 1, math.inf
        if hardcoded:
            return int(self._groundTruthLidar(point, column)), math.inf
        # END HARDCODED

        if self.number[column] == 1:
            return 0, math.inf

        if projection:
            point = point.projection(parameter(space, self.space))
        nearest, d = operations.nearestNeighborsFromDataContiguous(self.data[column, 1:self.number[column]], point.npPlain())
        if d[0] > self.space.maxDistance * self.CENTER_RADIUS:
            return 0, math.inf
        return nearest[0] + 1, d[0]

    def findAreaForOneColumn(self, point, column, space=None):
        return self._findAreaForOneColumn(point, column, space, True)[0]

    def findAreaForAllColumns(self, point, space=None):
        point = point.projection(parameter(space, self.space))
        return [self._findAreaForOneColumn(point, column, space)[0] for column in range(self.dim)]
    
    def findAreaForAllColumnsDistances(self, point, space=None):
        point = point.projection(parameter(space, self.space))
        r = [self._findAreaForOneColumn(point, column, space) for column in range(self.dim)]
        return [item[0] for item in r], [item[1] for item in r]
    
    def columns(self, point, space=None):
        point = point.projection(parameter(space, self.space))
        # relevances = np.array([False, True])
        # return np.array([relevances[self._findAreaForOneColumn(point, column, space, hardcoded=True)[0]] for column in range(self.dim)])
        return np.array([self.relevances[column, self._findAreaForOneColumn(point, column, space)[0]] for column in range(self.dim)])
    
    def columnsAreas(self, areas):
        return np.array([self.relevances[column, area] for column, area in enumerate(areas)])
    
    def findAllAreasForColumn(self, column):
        if self.number[column] == 1:
            return [0] * self.space.getData().shape[0]

        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.data[column, 1:self.number[column]])
        data = self.space.getData()
        dists, indices = nbrs.kneighbors(data)
        dists, indices = dists.flatten(), indices.flatten() + 1
        indices[dists > self.space.maxDistance * self.CENTER_RADIUS] = 0
        return indices
    
    def update(self, point):
        self.addPoint(point)
        self.addPoint(self.space.randomPoint())

    def addPoint(self, point, test=False):
        # if not test:
        #     return
        if self.space.number < self.MINIMUM_POINTS:
            return

        noiseMargin = 0.002

        probality = max(0.2, np.exp(-self.stability * 0.05))
        if random.uniform(0, 1) < probality:
            self.stability += 1

            point = point.projection(self.space)
            nearest, _ = self.space.nearest(point, n=self.NN_NUMBER)

            areas, distances = self.findAreaForAllColumnsDistances(point)

            currentColumns = self.columnsAreas(areas)

            errors = self.model.forwardErrors(onlyIds=nearest, precise=True, contextColumns=currentColumns)
            c = self.model.computeCompetence(np.mean(errors))
            # c = self.model.competence(onlyIds=nearest, precise=True, contextColumns=currentColumns)

            # print(columns)
            columns = np.arange(self.evaluatedSpace.dim)
            bestAdd = ()
            bestDel = ()
            checkDeletion = False
            if np.any(currentColumns) and random.uniform(0, 1) < 0.5:
                checkDeletion = True

            # print(currentColumns)
            for column in columns:
                if currentColumns[column] == checkDeletion:
                    # print(nearest)
                    # print(self.space.getData(nearest))
                    # print(self.model.outcomeSpace.getData(nearest))
                    # print(currentColumns)
                    # print(self.evaluatedSpace.getData(nearest)[:, currentColumns])
                    # if c is None:

                    # print(c)

                    newColumns = np.copy(currentColumns)
                    newColumns[column] = not newColumns[column]
                    # newColumns[(column + 1) % 8] = not newColumns[(column + 1) % 8]
                    # newColumns[(column - 1) % 8] = not newColumns[(column - 1) % 8]

                    # newColumnsOverwrite = None
                    newColumnsOverwrite = np.full_like(currentColumns, True)
                    newColumnsOverwrite[column] = False
                    # print(newColumns)
                    # print(self.evaluatedSpace.getData(nearest)[:, newColumns])
                    newErrors = self.model.forwardErrors(onlyIds=nearest, contextColumns=newColumns, precise=True, contextColumnsOverwrite=newColumnsOverwrite)
                    nc = self.model.computeCompetence(np.mean(newErrors))
                    # nc = self.model.competence(onlyIds=nearest, contextColumns=newColumns, precise=True, contextColumnsOverwrite=newColumnsOverwrite)

                    progress = -(newErrors - errors)
                    progress = np.sign(progress) * (np.clip(np.abs(progress) - noiseMargin, 0., None))

                    pointProgress = progress[0]
                    meanProgress = np.mean(progress)
                    meanRegression = np.percentile(progress, 10)
                    pc = nc - c
                    progressValue = meanProgress + meanRegression + pc * 0.02

                    # should = self._groundTruthLidar(point, column)
                    if test > 1:
                        print(f'{column}: ={newColumns[column]} ==== {c}+{progressValue} ({meanRegression} {meanRegression} {pc})')
                        for error, nerror, pr, ptx, ctx in zip(errors, newErrors, progress, self.space.getData(nearest), self.evaluatedSpace.getData(nearest)):
                            print(f'{ptx} | {ctx[newColumns]} -> {error:.3f}->{nerror:.3f} {pr:.3f}')
                    # print(progressValue)
                    # addition = newColumns[column]

                    if not checkDeletion and pointProgress >= self.THRESHOLD_ADD_POINT and progressValue >= self.THRESHOLD_ADD and (not bestAdd or progressValue > bestAdd[0]):
                        bestAdd = (progressValue, column, newColumns)
                    elif checkDeletion and pointProgress >= self.THRESHOLD_DEL_POINT and progressValue >= self.THRESHOLD_DEL and (not bestDel or progressValue > bestDel[0]):
                        bestDel = (progressValue, column, newColumns)

                    # registerColumns = None
                    # if (addition and p >= self.THRESHOLD_ADD) or (not addition and p >= self.THRESHOLD_DEL):
                    #     registerColumns = newColumns
                    # # else:
                    # #     registerColumns = currentColumns
                    # if registerColumns is not None:
                    #     canCreateNew = distances[column] >= self.space.maxDistance * self.MIN_DISTANCE_CENTERS
                    #     print(f'Create new columns for {column}: {registerColumns[column]} around {point.plain()} {p}')

                    #     if canCreateNew:
                    #         self._addArea(column, point, registerColumns[column])
                    #     else:
                    #         self._updateArea(column, point, registerColumns[column], areas[column])

                # if (bestAdd or bestDel) and not fullCompAllFalse:
                #     columnsFalse = np.full(self.evaluatedSpace.dim, False)
                #     fullCompAllFalse = self.model.competence(
                #         precise=True, contextColumns=columnsFalse)

                #     columnsTrue = np.full(self.evaluatedSpace.dim, True)
                #     fullCompAllTrue = self.model.competence(
                #         precise=True, contextColumns=columnsTrue)
                    
                #     bestFullComp = max(fullCompAllFalse, fullCompAllTrue)

            for best, deletion, verb in ((bestAdd, False, 'add'),): #, (bestDel, True, 'del')):
                if best:
                    self.stability = 0
                    progressValue, column, newColumns = best
                    # print(f'Should {verb} context column {column} (+{progressValue:.2f}) around {point}')

                    canCreateNew = distances[column] >= self.space.maxDistance * self.MIN_DISTANCE_CENTERS
                    # print(f'Create new columns for {column}: {newColumns[column]} around {point.plain()} +{progressValue} new: {canCreateNew}')

                    # should = self._groundTruthLidar(point, column)
                    # if not should:
                    #     print(f'!! Invalid choice of column {column} for {point}#{nearest[0]} +{progressValue}!\n{self.columns(point)}')
                    #     return 1, 0

                    if canCreateNew:
                        self._addArea(column, point, newColumns[column])
                    else:
                        self._updateArea(column, point, newColumns[column], areas[column])
            
            # return 0, 1 if bestAdd else 0
            # if not bestAdd:
            #     print(f'>> No new column for {point}#{nearest[0]}!\n{self.columns(point)}')


            # for best, deletion, verb in ((bestAdd, False, 'add'), (bestDel, False, 'del')):
            #     if best:
            #         p, column, newColumns = best
            #         print(f'Should {verb} context column {column} ({c:.2f}+{p:.2f}) around {point}')
            #         newArea = None

            #         area = areas[column]
            #         # fullComp = self.model.competence(precise=True)
            #         canCreateNew = distances[column] >= self.space.maxDistance * self.MIN_DISTANCE_CENTERS
            #         if not area.default and area.attempt(currentColumns, newColumns, deletion, canCreateNew):
            #             print('Updating current area')
            #             # previousUsed = area.used
            #             area.used = newColumns[column]
            #         else:
            #             if not canCreateNew:
            #                 print('To close!')
            #                 continue
            #             print('Creating a new area')
            #             newArea = ContextArea(self, point, column, newColumns[column])
            #             self._addArea(newArea)

            #         # newFullComp = self.model.competence(precise=True)
            #         # print(f'Variation: {newFullComp - fullComp}')
            #         # if newFullComp < fullComp * self.THRESHOLD_VALID:  # bestFullComp - self.THRESHOLD_RESET:
            #         #     print('Reverting...')
            #         #     if newArea:
            #         #         self._removeArea(newArea)
            #         #     else:
            #         #         area.used = previousUsed
            #         #     currentColumns = newColumns
            #         # else:
            #         area.stability = 0
    
    # Visual
    def visualizeAreaColumn(self, column=0, options={}):
        g = Graph(title=f'Context areas from {self.space} for column {column}', options=options)

        # All points
        areas = self.findAllAreasForColumn(column)
        points = self.space.getData()

        for relevant in (True, False):
            data = points[np.argwhere(self.relevances[column, areas] == relevant).flatten(), :]
            if len(data) > 0:
                g.scatter(data, label=relevant, color=('orange' if relevant else 'gray'))

        # Centroids
        for (relevant, name, color) in ((True, 'Relevant', 'red'), (False, 'Non relevant', 'black')):
            data = np.argwhere(self.relevances[column, :self.number[column]] == relevant).flatten()
            if not relevant:
                data = data[1:]
            if len(data) > 0:
                g.scatter(self.data[column, data, :], label=f'{name} centers', color=color, marker='x')
        return g

    def visualizeAreaColumns(self, options={}):
        g = Graph(title=f'Context areas from {self.space}', options=options)

        points = self.space.getData()
        offsetWidth = self.model.actionSpace.maxDistance * 0.01
        offsetN = np.math.ceil(np.math.sqrt(self.dim))

        for column in range(self.dim):
            offset = np.array([offsetWidth * (-offsetN / 2 + column % offsetN),
                               offsetWidth * (-offsetN / 2 + column // offsetN)])

            areas = self.findAllAreasForColumn(column)
            data = points[np.argwhere(self.relevances[column, areas]).flatten(), :] + offset
            if len(data) > 0:
                g.scatter(data, label=column, alpha=0.9, marker='.')
        return g


class ContextSpatializationAreas(Serializable):
    MAX_AREAS = 100
    NN_NUMBER = 20
    MINIMUM_POINTS = 100
    THRESHOLD_ADD = 0.05
    THRESHOLD_DEL = 0.
    THRESHOLD_RESET = 0.05
    THRESHOLD_VALID = 0.5
    THRESHOLD_CANT_CREATE = 0.1
    MIN_DISTANCE_CENTERS = 0.05

    def __init__(self, model, space, boolean=False):
        self.model = model
        self.space = space
        self.evaluatedSpace = self.model.contextSpace
        self.dim = self.evaluatedSpace.dim
        self.boolean = boolean

        self.defaultAreas = [ContextArea(self, self.space.zero(), i, False, default=True) for i in range(self.dim)]
        self.resetAreas()

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['areas', 'defaultAreas'])
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            raise Exception('No full deserializer is available for this class')
        return super()._deserialize(dict_, serializer, obj)

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        serializer = serializer.clone(values={'.area.manager': self})

        for areasDict in dict_.get('areas', []):
            for areaDict in areasDict:
                area = serializer.deserialize(areaDict)
                self._addArea(area)
        self.defaultAreas = serializer.deserialize(dict_.get('defaultAreas', []))

    @property
    def dataset(self):
        return self.model.dataset
    
    # def continueFrom(self, cs):
    #     for area in cs.areas:
    #         self._addArea(area.cloneTo(self))

    def _addArea(self, area):
        column = area.column
        if len(self.areas[column]) >= self.MAX_AREAS:
            return

        self.centers[column, len(self.areas[column]), :] = area.center.npPlain()
        self.areas[column].append(area)

        self.findAllPointAreas()
        self.model.invalidateCompetences()

    def _removeArea(self, area):
        column = area.column
        index = self.areas[column].index(area)
        self.centers[column, index:-1, :] = self.centers[column, index+1:, :]
        self.areas[column].remove(area)

        self.findAllPointAreas()
        self.model.invalidateCompetences()

    def resetAreas(self):
        self.areas = [[] for _ in range(self.dim)]
        self.centers = np.zeros((self.dim, self.MAX_AREAS, self.space.dim))

        self.model.invalidateCompetences()

    def allTrue(self, value=True):
        self.resetAreas()
        for column in range(self.dim):
            self._addArea(ContextArea(self, self.space.zero(), column, True))
    
    def allFalse(self):
        self.resetAreas()
    
    def _findAreaForOneColumn(self, point, column, space, projection=False):
        if not self.areas[column]:
            return self.defaultAreas[column], math.inf

        if projection:
            point = point.projection(parameter(space, self.space))
        nearest, d = operations.nearestNeighborsFromDataContiguous(self.centers[column, :len(self.areas[column])], point.npPlain())
        return self.areas[column][nearest[0]], d[0]

    def findAreaForOneColumn(self, point, column, space=None):
        return self._findAreaForOneColumn(point, column, space, True)[0]

    def findAreaForAllColumns(self, point, space=None):
        point = point.projection(parameter(space, self.space))
        return [self._findAreaForOneColumn(point, column, space)[0] for column in range(self.dim)]
    
    def findAreaForAllColumnsDistances(self, point, space=None):
        point = point.projection(parameter(space, self.space))
        r = [self._findAreaForOneColumn(point, column, space) for column in range(self.dim)]
        return [item[0] for item in r], [item[1] for item in r]
    
    def columns(self, goal, space=None):
        areas = self.findAreaForAllColumns(goal, space)
        return np.array([area.used for area in areas])

    def columnsAreas(self, areas):
        return np.array([area.used for area in areas])
    
    def findAllPointAreas(self):
        for column in range(self.dim):
            if self.areas[column]:
                nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(self.centers[column, :len(self.areas[column])])
                data = self.space.getData()
                _, indices = nbrs.kneighbors(data)
                indices = indices.flatten()
                for i, area in enumerate(self.areas[column]):
                    area.ids = self.space.getIds()[np.argwhere(indices == i).flatten()]
    
    def update(self, point):
        self.addPoint(point)
        self.addPoint(self.space.randomPoint())

    def addPoint(self, point):
        # return
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

        areas, distances = self.findAreaForAllColumnsDistances(point)

        # dims = np.arange(self.evaluatedSpace.dim)
        # np.random.shuffle(areas)
        # n = random.randint(1, min(1 + int(self.evaluatedSpace.dim * 0.4), 3))
        # selectedAreas = areas[:n]
        selectedAreas = areas

        # print(areas)
        for area in selectedAreas:
            area.addPoint(id_)
            area.stability += 1
            column = area.column

            probality = 2.  # max(0.05, np.exp(-stability * 0.1))
            # print(self.model, stability, probality)

            # fullCompAllFalse = None
            # fullCompAllTrue = None
            # if area and random.uniform(0, 1) < probality * 0.1:
            #     fullComp = self.model.competence(precise=True)

            #     columnsFalse = np.full(self.evaluatedSpace.dim, False)
            #     fullCompAllFalse = self.model.competence(precise=True, contextColumns=columnsFalse)

            #     columnsTrue = np.full(self.evaluatedSpace.dim, True)
            #     fullCompAllTrue = self.model.competence(precise=True, contextColumns=columnsTrue)

            #     bestFullComp = max(fullCompAllFalse, fullCompAllTrue)
            #     columns = columnsTrue if fullCompAllTrue > fullCompAllFalse else columnsFalse

            #     if fullComp < bestFullComp - self.THRESHOLD_RESET * 2:
            #         # print('Reset all areas!')
            #         self.resetAreas()
            #         area = ContextArea(self, point, columns)
            #         self._addArea(area)

            c = None
            if random.uniform(0, 1) < probality:
                # print(nearest)
                # print(self.space.getData(nearest))
                # print(self.model.outcomeSpace.getData(nearest))
                # print(currentColumns)
                # print(self.evaluatedSpace.getData(nearest)[:, currentColumns])
                if c is None:
                    currentColumns = self.columnsAreas(areas)
                    c = self.model.competence(onlyIds=nearest, precise=True)
                # print(c)

                columns = np.copy(currentColumns)
                columns[column] = not columns[column]
                # print(columns)
                # print(self.evaluatedSpace.getData(nearest)[:, columns])
                nc = self.model.competence(onlyIds=nearest, contextColumns=columns, precise=True)
                p = nc - c
                # print(p)
                if columns[column] and p >= self.THRESHOLD_ADD and (not bestAdd or p > bestAdd[0]):
                    bestAdd = (p, column, columns)
                if not columns[column] and p >= self.THRESHOLD_DEL and (not bestDel or p > bestDel[0]):
                    bestDel = (p, column, columns)

            # if (bestAdd or bestDel) and not fullCompAllFalse:
            #     columnsFalse = np.full(self.evaluatedSpace.dim, False)
            #     fullCompAllFalse = self.model.competence(
            #         precise=True, contextColumns=columnsFalse)

            #     columnsTrue = np.full(self.evaluatedSpace.dim, True)
            #     fullCompAllTrue = self.model.competence(
            #         precise=True, contextColumns=columnsTrue)
                
            #     bestFullComp = max(fullCompAllFalse, fullCompAllTrue)

        for best, deletion, verb in ((bestAdd, False, 'add'), (bestDel, False, 'del')):
            if best:
                p, column, newColumns = best
                print(f'Should {verb} context column {column} ({c:.2f}+{p:.2f}) around {point}')
                newArea = None

                area = areas[column]
                # fullComp = self.model.competence(precise=True)
                canCreateNew = distances[column] >= self.space.maxDistance * self.MIN_DISTANCE_CENTERS
                if not area.default and area.attempt(currentColumns, newColumns, deletion, canCreateNew):
                    print('Updating current area')
                    # previousUsed = area.used
                    area.used = newColumns[column]
                else:
                    if not canCreateNew:
                        print('To close!')
                        continue
                    print('Creating a new area')
                    newArea = ContextArea(self, point, column, newColumns[column])
                    self._addArea(newArea)

                # newFullComp = self.model.competence(precise=True)
                # print(f'Variation: {newFullComp - fullComp}')
                # if newFullComp < fullComp * self.THRESHOLD_VALID:  # bestFullComp - self.THRESHOLD_RESET:
                #     print('Reverting...')
                #     if newArea:
                #         self._removeArea(newArea)
                #     else:
                #         area.used = previousUsed
                #     currentColumns = newColumns
                # else:
                area.stability = 0
    
    # Visual
    def visualizeAreaColumn(self, column=0, options={}):
        g = Graph(title=f'Context areas from {self.space} for column {column}', options=options)
        areas = {}
        for area in self.areas[column]:
            if area.used in areas.keys():
                areas[area.used] = np.vstack((areas[area.used], self.space.getData(area.ids)))
            else:
                areas[area.used] = self.space.getData(area.ids)
        for used, data in areas.items():
            g.scatter(data, label=used, color=('orange' if used else 'gray'))
        g.scatter(self.centers[column, :len(self.areas[column]), :], label='Center', color='red', marker='x')
        return g

    def visualizeAreaColumns(self, options={}):
        g = Graph(title=f'Context areas from {self.space}', options=options)
        areas = {}
        allAreas = [b for a in self.areas for b in a]

        offsetWidth = self.model.actionSpace.maxDistance * 0.01
        offsetN = np.math.ceil(np.math.sqrt(self.dim))
        for area in allAreas:
            if area.used:
                offset = np.array([offsetWidth * (-offsetN / 2 + area.column % offsetN),
                                   offsetWidth * (-offsetN / 2 + area.column // offsetN)])
                if area.column in areas.keys():
                    areas[area.column] = np.vstack((areas[area.column], self.space.getData(area.ids) + offset))
                else:
                    areas[area.column] = self.space.getData(area.ids) + offset
        for column, data in areas.items():
            g.scatter(data, label=column, alpha=0.9, marker='.')
        return g


class ContextArea(Serializable):
    def __init__(self, manager, center, column, used, default=False):
        self.manager = manager
        self.center = center
        self.column = column
        self.used = used
        self.ids = np.array([])
        self.stability = 0
        self.default = default
    
    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['center', 'column', 'stability', 'default'])
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls(serializer.get('.area.manager'),
                      serializer.deserialize(dict_.get('center')),
                      serializer.deserialize(dict_.get('column')),
                      serializer.deserialize(dict_.get('used')),
                      serializer.deserialize(dict_.get('default'))
                      )
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
        if not self.default:
            self.ids = np.append(self.ids, id_)
    
    def attempt(self, columns, newColumns, deletion, canCreateNew, precise=True):
        c = self.manager.model.competence(
            onlyIds=self.ids, contextColumns=columns, precise=precise)
        nc = self.manager.model.competence(
            onlyIds=self.ids, contextColumns=newColumns, precise=precise)
        print(f'Attempt on current region: p={nc - c}')
        return nc - c > (ContextSpatialization.THRESHOLD_DEL if deletion else ContextSpatialization.THRESHOLD_ADD) - \
                (0. if canCreateNew else ContextSpatialization.THRESHOLD_CANT_CREATE)


    def __repr__(self):
        if self not in self.manager.areas:
            return f'Area centered on {self.center}'
        return f'Area {self.manager.areas.index(self)} centered on {self.center}'
