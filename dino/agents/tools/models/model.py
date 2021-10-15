import sys
import copy
import math
import random
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean

# import matplotlib as plt

# from ..utils.logging import DataEventHistory, DataEventKind
# from ..utils.io import getVisual, plotData, visualize, concat
# from ..utils.serializer import Serializable, Serializer
# from ..utils.maths import multivariateRegression, multivariateRegressionError, multivariateRegression_vector, popn
from exlab.utils.io import parameter
from exlab.interface.serializer import Serializable


from dino.data.data import *
from dino.data.space import SpaceKind
# from ..data.abstract import *
from dino.data.path import ActionNotFound
from dino.data.contextarea import ContextSpatialization


'''

'''


class Model(Serializable):
    number = 0
    ALMOST_ZERO_FACTOR = 0.002
    PRECISION_LEARNING_RATE = 0.01
    PRECISION_MULTIPLIER = 2.
    PRECISION_GOAL_NUMBER = 25

    def __init__(self, dataset, actionSpace, outcomeSpace, contextSpace=[], restrictionIds=None, model=None,
                 register=True, metaData={}):
        self.id = Model.number
        Model.number += 1

        self.dataset = dataset
        self.enabled = True

        self.allowCompetenceCaching = True
        self.invalidateCompetences()

        self.createdSince = -1
        self.lowCompetenceSince = -1
        self.evaluations = {}
        self.attemptedContextSpaces = {}
        self.precision = [-1., -1.]
        self.precisionPerGoalNorm = np.zeros(self.PRECISION_GOAL_NUMBER)
        self.precisionPerGoalNormNumber = np.zeros(self.PRECISION_GOAL_NUMBER)
        self.limitMoves = 1.

        self.restrictionIds = restrictionIds
        # self.nonDuplicateLastId = -1
        # self.nonDuplicateActionIds = []
        # self.nonDuplicateOutcomeIds = []

        self.actionSpace = dataset.multiColSpace(actionSpace)
        self.outcomeSpace = dataset.multiColSpace(outcomeSpace)
        self.contextSpace = dataset.multiColSpace(contextSpace)
        self.contextSpace = self.contextSpace.convertTo(self.dataset, kind=SpaceKind.PRE)

        self.actionContextSpace = dataset.multiColSpace(
            [self.actionSpace, self.contextSpace], weight=0.5)
        self.outcomeContextSpace = dataset.multiColSpace(
            [self.outcomeSpace, self.contextSpace], weight=0.5)

        self.contextSpacialization = None
        if self.contextSpace:
            self.contextSpacialization = [ContextSpatialization(self, self.actionSpace)] #, ContextSpatialization(self, self.contextSpace)]

        # self.spacesHistory = DataEventHistory()

        if register:
            self.dataset.registerModel(self)

    def __repr__(self):
        disabled = '❌' if not self.enabled else ''
        return f'{disabled}Model{self.__class__.__name__}({self.actionSpace} | {self.contextSpace} => {self.outcomeSpace})'
    
    def _sid(self, serializer):
        return serializer.serialize(self, foreigns=['dataset', 'actionSpace', 'outcomeSpace', 'contextSpace'], reference=True)

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['enabled', 'createdSince', 'lowCompetenceSince', 'contextSpacialization', 'precision'],
            foreigns=['dataset', 'actionSpace', 'outcomeSpace', 'contextSpace'])
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls._findOrCreate(serializer.deserialize(dict_.get('dataset')),
                    serializer.deserialize(dict_.get('actionSpace')),
                    serializer.deserialize(dict_.get('outcomeSpace')),
                    serializer.deserialize(dict_.get('contextSpace')),
                    dict_.get('metaData', {}))
        return super()._deserialize(dict_, serializer, obj)

    @classmethod
    def _findOrCreate(cls, dataset, actionSpace, outcomeSpace, contextSpace=[], metaData={}):
        obj = None
        if obj is None:
            obj = cls(dataset, actionSpace, outcomeSpace, contextSpace, metaData=metaData)
        return obj

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        for attr in ['enabled', 'createdSince', 'lowCompetenceSince', 'precision']:
            setattr(self, attr, dict_.get(attr))
        for i, csd in enumerate(dict_.get('contextSpacialization', [])):
            if i < len(self.contextSpacialization):
                serializer.deserialize(csd, obj=self.contextSpacialization[i])

    # @classmethod
    # def _deserialize(cls, dict_, dataset, spaces, loadResults=True, options={}, obj=None):
    #     a = [next(i for i in spaces if i.name == name or i.id == name)
    #          for name in dict_['actions']]
    #     y = [next(i for i in spaces if i.name == name or i.id == name)
    #          for name in dict_['outcomes']]
    #     c = [next(i for i in spaces if i.name == name or i.id == name)
    #          for name in dict_.get('context', [])]
    #     obj = cls(dataset, a, y, c)
    #     return obj

    # def actualOutcomeContextSpace(self, contextColumns):
    #     if not np.any(contextColumns):
    #         return self.outcomeSpace
    #     return self.outcomeContextSpace
    
    # def actualActionContextSpace(self, contextColumns):
    #     if not np.any(contextColumns):
    #         return self.actionSpace
    #     return self.actionContextSpace

    @classmethod
    def adaptiveEpisode(cls, adaptiveModelManager, events):
        pass

    def justCreated(self):
        pass

    @property
    def duration(self):
        if self.createdSince == -1:
            return 0
        return self.dataset.learner.iteration - self.createdSince
    
    def matches(self, model, ignoreContext=False):
        return (self.actionSpace == model.actionSpace and
                self.outcomeSpace == model.outcomeSpace and
                (ignoreContext or self.contextSpace == model.contextSpace or (not self.contextSpace and not model.contextSpace)))
    
    def includes(self, model, ignoreContext=False):
        return (self.actionSpace.includes(model.actionSpace) and
                self.outcomeSpace.includes(model.outcomeSpace) and
                (ignoreContext or (not self.contextSpace and not model.contextSpace) or self.contextSpace.includes(model.contextSpace)))

    def update(self):
        pass

    def metaData(self):
        return None

    def continueFrom(self, previousModel):
        self.id = previousModel.id
        previousModel.id = -1

        self.createdSince = previousModel.createdSince
        self.lowCompetenceSince = previousModel.lowCompetenceSince
        self.evaluations = previousModel.evaluations

        # if self.contextSpacialization and previousModel.contextSpacialization:
        #     self.contextSpacialization[0].continueFrom(
        #         previousModel.contextSpacialization[0])

        # previousModel.spacesHistory.extend(self.spacesHistory)
        # self.spacesHistory = previousModel.spacesHistory
    
    def findActionFromEvent(self, event):
        if self.isCoveredByActionSpaces(event.actions.space):
            return event.actions.projection(self.actionSpace)
        elif self.isCoveredByActionSpaces(event.primitiveActions.space):
            return event.primitiveActions.projection(self.actionSpace)
        else:
            return event.outcomes.projection(self.actionSpace)

    def pointAdded(self, event):
        if self.contextSpacialization:
            # print(event.allActions)
            # outcome = event.outcomes.projection(self.outcomeSpace)
            self.contextSpacialization[0].update(event.allActions)

            # context = event.context.projection(self.contextSpace)
            # self.contextSpacialization[1].addPoint(context, event.iteration)

    def updatePrecision(self, success, distanceToGoal, weight=1, index=0):
        # oldp = self.precision[index]
        precision = min(distanceToGoal * self.PRECISION_MULTIPLIER, self.outcomeSpace.maxDistance * 0.5)
        if self.precision[index] < 0:
            self.precision[index] = self.outcomeSpace.maxDistance * 0.03
        learningRate = self.PRECISION_LEARNING_RATE * weight
        if distanceToGoal < self.precision[index]:
            if not success:
                return
            learningRate *= 2
        self.precision[index] = self.precision[index] * (1 - learningRate) + (precision - self.precision[index]) * learningRate
        if index == 1 and self.precision[index] < self.precision[0]:
            self.precision[index] = self.precision[0]
        self.precision[index] = min(max(self.precision[index], self.outcomeSpace.maxDistance * 0.005), self.outcomeSpace.maxDistance * 0.5)
        # print(f'#{index} from {oldp} to {self.precision[index]} // {distanceToGoal} // {self}')
    
    def getPrecision(self, ratio=0.01, multiplier=1):
        if self.precision[0] < 0:
            return self.outcomeSpace.maxDistance * ratio
        return self.precision[0] * multiplier
    
    def updatePrecisionPerGoal(self, goalMove, error):
        index = int(goalMove.norm() / self.outcomeSpace.maxDistance * self.PRECISION_GOAL_NUMBER)
        precision = min(error, self.outcomeSpace.maxDistance * 0.5)
        if index >= self.PRECISION_GOAL_NUMBER:
            return
        # learningRate = self.PRECISION_LEARNING_RATE * 5
        self.precisionPerGoalNormNumber[index] = min(self.precisionPerGoalNormNumber[index] + 1, 100)
        self.precisionPerGoalNorm[index] = (
            (self.precisionPerGoalNorm[index] * (self.precisionPerGoalNormNumber[index] - 1)) + precision) / self.precisionPerGoalNormNumber[index]
        
        # print(f'UPDATING {self} {goalMove.norm()} => {index}: {error}')
    
    def explorable(self):
        return self.precision[0] >= 0.
        #  and self.precision[0] <= self.outcomeSpace.maxDistance * 0.25
    
    def maxDistance(self, outcome):
        return (self.outcomeSpace.maxDistance / 4 if self.outcomeSpace.maxDistance != 0 else 1.) + outcome.norm() / 2

    # Context
    def hasContext(self, contextSpace, contextColumns):
        return contextSpace and (contextColumns is None or np.any(contextColumns))

    def contextColumns(self, contextColumns, action, context):
        if contextColumns is not None:
            return contextColumns
        if not self.contextSpacialization:
            return None
        return self.contextSpacialization[0].columns(action)# & self.contextSpacialization[1].columns(context)

    def multiContextColumns(self, contextColumns, space, context):
        if contextColumns is None or not np.any(contextColumns) or context is None:
            return None
        indices = space.columnsFor(self.contextSpace)
        cols = np.full(space.dim, True)
        cols[indices] = contextColumns
        return cols
    
    def retrieveContext(self, context, entity=None):
        return context.convertTo(self.dataset, kind=SpaceKind.PRE).projection(self.contextSpace, entity=entity)

    # Operations
    def forward(self, action: Action, context: Observation = None, contextColumns=None, ignoreFirst=False, entity=None):
        value, error = self.npForward(action, context, contextColumns=contextColumns, ignoreFirst=ignoreFirst, entity=entity)
        if value is None or np.isnan(np.sum(value)):
            return None, 1
        return Data(self.outcomeSpace.applyTo(entity), value.tolist()).setRelative(True), error

    def npForward(self, action: Action, context: Observation = None, contextColumns=None, ignoreFirst=False, entity=None):
        raise NotImplementedError()

    def inverse(self, goal: Goal, context: Observation = None, contextColumns=None, entity=None):
        raise NotImplementedError()

    def bestLocality(self, goal: Goal, context: Observation = None, contextColumns=None, entity=None, dontMove=[]):
        raise NotImplementedError()

    def computeCompetence(self, error):
        error = min(error, 2)
        return max(0, min(1., (1. - error ** 2) / np.exp((error * 20))))  # ** 3
    
    # def computeCompetence(self, error, distanceGoal=0):
    #     distanceGoal = min(distanceGoal, 1.)
    #     return max(0, min(1, (1. - distanceGoal - error) / np.exp((error * 4.15) ** 3)))

    def getIds(self):
        return self.findSharedIds(self.outcomeSpace, self.actionSpace, self.contextSpace, restrictionIds=self.restrictionIds)
    
    # def updateNonDuplicateIds(self):
    #     number = self.actionSpace.number
    #     if self.nonDuplicateLastActionId < number:
    #         self.actionSpace.findDuplicates(self.nonDuplicateActionIds)

    #         self.nonDuplicateLastActionId < number

    #     self.nonDuplicateLastId = id_
    #     self.nonDuplicateActionIds = []
    #     self.nonDuplicateOutcomeIds = []
    
    @staticmethod
    def findSharedIds(self, *spaces, restrictionIds):
        assert len(spaces) > 0, 'At least 1 space is required!'

        spaces = list(spaces)
        ids = spaces.pop(0).getIds()
        if restrictionIds:
            ids = np.intersect1d(ids, restrictionIds)
        for space in spaces:
            ids = np.intersect1d(ids, space.getIds())
        return ids

    def competence(self, precise=False, onlyIds=None, contextColumns=None, contextColumnsOverwrite=None):
        if onlyIds is None and contextColumns is None:
            if self._lastCompetencePrecise:
                return self._lastCompetencePrecise
            elif not precise and self._lastCompetence:
                return self._lastCompetence

        c = self.computeCompetence(self.std(precise=precise, onlyIds=onlyIds, contextColumns=contextColumns, contextColumnsOverwrite=contextColumnsOverwrite))

        if onlyIds is None and contextColumns is None:
            if precise:
                self._lastCompetencePrecise = c
            else:
                self._lastCompetence = c
        return c

    def competenceData(self, actions, outcomes, contexts=None, precise=False, contextColumns=None):
        return self.computeCompetence(self.stdData(actions, outcomes, contexts, precise=precise, contextColumns=contextColumns))

    def invalidateCompetences(self):
        self._lastCompetence = None
        self._lastCompetencePrecise = None
    
    def performant(self, competence, duration):
        return self.competence(precise=True) >= competence and self.duration >= duration

    # data=None, context: Observation = None,
    def std(self, precise=False, onlyIds=None, contextColumns=None, contextColumnsOverwrite=None):
        errorsZeros, _, _ = self._errorForwardAll(
            precise=precise, almostZeros=True, onlyIds=onlyIds, contextColumns=contextColumns, contextColumnsOverwrite=contextColumnsOverwrite)
        errorZeros = np.mean(errorsZeros) if len(errorsZeros) > 0 else 0.

        errorsNonZeros, _, _ = self._errorForwardAll(
            precise=precise, almostZeros=False, onlyIds=onlyIds, contextColumns=contextColumns, contextColumnsOverwrite=contextColumnsOverwrite)
        errorNonZeros = np.mean(errorsNonZeros) if len(errorsNonZeros) > 0 else 0.

        #np.percentile(errorsZeros, 65)
        # print('---', errorZeros, errorNonZeros)

        return 1 - (1 - errorNonZeros) * (1 - errorZeros)
    
    def stdData(self, actions, outcomes, contexts=None, precise=False, onlyIds=None, contextColumns=None):
        errors, _ = self._errorForwardDataAll(
            actions, outcomes, contexts, precise=precise, contextColumns=contextColumns)
        if len(errors) == 0:
            return 0.
        return np.mean(errors)
    
    # def variance(self, data=None, context: Observation = None, precise=False, contextColumns=None):
    #     return self.std(data=data, context=context, precise=precise, contextColumns=contextColumns) ** 2

    def forwardErrors(self, precise=False, almostZeros=None, onlyIds=None, contextColumns=None, contextColumnsOverwrite=None, sortIds=False):
        errors, _, ids = self._errorForwardAll(precise=precise, almostZeros=almostZeros, onlyIds=onlyIds,
                                            contextColumns=contextColumns, contextColumnsOverwrite=contextColumnsOverwrite)
        if sortIds:
            order = np.argsort(-errors)
            distanceMax = (self.outcomeSpace.maxDistance / 4 if self.outcomeSpace.maxDistance != 0 else 1.)
            trueErrors = errors * distanceMax
            return errors[order], ids[order], trueErrors[order]
        return errors

    def forwardError(self, eventId, contextColumns=None, contextColumnsOverwrite=None):
        action = self.actionSpace.getPoint(eventId)[0]
        outcome = self.outcomeSpace.getPoint(eventId)[0]
        context = self.contextSpace.getPoint(eventId)[0] if self.contextSpace else None

        return self._errorForwardData(action, outcome, context, contextColumns=contextColumns, contextColumnsOverwrite=contextColumnsOverwrite, eventId=eventId)
    
    def _errorForwardData(self, action, outcome, context=None, contextColumns=None, contextColumnsOverwrite=None, eventId=None):
        if not self.contextSpace:
            context = None
        if contextColumns is not None and contextColumnsOverwrite is not None:
            contextColumns[contextColumnsOverwrite] = self.contextSpacialization[0].columns(action)[contextColumnsOverwrite]
        # print(f'{self.actionSpace.getPoint(eventId)[0]} + {self.contextSpace.getPoint(eventId)[0]} -> {self.outcomeSpace.getPoint(eventId)[0]}')

        # contextColumns = self.contextColumns(contextColumns, action, context)

        # context = context if self.hasContext(
        #     self.contextSpace, contextColumns) else None

        # if context is None:
        #     contextColumns = None

        # print(contextColumns)
        outcomeEstimated, linearError = self.forward(action, context, contextColumns=contextColumns, ignoreFirst=False)
        # acPlain = action.extends(context).npPlain()
        # outcomeEstimated = self.outcomeSpace.point(self.fnn.predict([acPlain])[0])
        if not outcomeEstimated:
            return 1.

        # print(outcome)
        # print(outcomeEstimated)
        # print(linearError)
        errorOutcome = outcomeEstimated.distanceTo(outcome) / self.maxDistance(outcome) + min(linearError, 1.) * 0.1

        errorOutcome = min(errorOutcome, 1.)
        # print(f'{action}, {outcome} ?= {outcomeEstimated} -> {errorOutcome}')
        # self.dataset.logger.info(f'{errorOutcome:.2f} #{eventId}: {action.plain()} + {context} -> {outcome.plain()} vs estimated {outcomeEstimated.plain()}')

        # if errorOutcome > 0.1 or linearError > 0.1:
        #     if context:
        #         context = context.plain()
        #     print(
        #         f'Failed {errorOutcome:.3f} {linearError:.3f} #{eventId}: {action.plain()} + {context} -> {outcome.plain()} vs estimated {outcomeEstimated.plain()}')
            # print('--- ERROR ---')
            # print(outcome)
            # print(outcomeEstimated)
            # print(errorOutcome)
            # print(context)
            # print(action)
            # print('---       ---')
        return errorOutcome, linearError

    def _errorForwardAll(self, precise=False, almostZeros=None, onlyIds=None, contextColumns=None, contextColumnsOverwrite=None):
        ids = parameter(onlyIds, self.getIds())
        if almostZeros is not None:
            data = self.outcomeSpace.getData(ids)
            indices = np.sum(np.abs(data), axis=1) > self.outcomeSpace.maxDistance * self.ALMOST_ZERO_FACTOR
            if almostZeros:
                indices = ~indices
            if np.sum(indices) == 0:
                return [], [], []
            ids = ids[indices]
        if not precise:
            number = min(80 if not almostZeros else 50, self.outcomeSpace.number // 10)
            ids_ = np.arange(len(ids))
            np.random.shuffle(ids_)
            ids = ids[ids_[:number]]
        errors = np.array([self.forwardError(id_, contextColumns=contextColumns, contextColumnsOverwrite=contextColumnsOverwrite) for id_ in ids]).T
        return errors[0], errors[1], ids

    def _errorForwardDataAll(self, actions, outcomes, contexts=None, precise=False, exceptAlmostZero=False, contextColumns=None):
        # data = np.hstack((actions, outcomes, contexts) if contexts is not None else (actions, outcomes))
        ids = np.arange(len(actions))
        if exceptAlmostZero:
            indices = np.sum(
                np.abs(outcomes), axis=1) > self.outcomeSpace.maxDistance * self.ALMOST_ZERO_FACTOR
            if np.sum(indices) == 0:
                return []
            ids = ids[indices]

        if not precise:
            number = min(100, self.outcomeSpace.number // 10)
            ids_ = np.arange(len(ids))
            np.random.shuffle(ids_)
            ids = ids[ids_[:number]]

        if contexts is not None:
            data = zip(actions, outcomes, contexts)
        else:
            data = zip(actions, outcomes)
        errors = np.array([self._errorForwardData(*d, contextColumns=contextColumns) for d in data]).T
        return errors[0], errors[1]

    # Space coverage
    def reachesSpaces(self, spaces):
        spaces = self.dataset.convertSpaces(spaces)
        return self.outcomeSpace.intersects(spaces)

    def __spaceCoverage(self, spaces, selfSpaces, covers=0):
        selfSpaces = set(selfSpaces.flatSpaces)
        if not isinstance(spaces, list):
            spaces = [spaces]
        spaces = self.dataset.convertSpaces(spaces)
        spaces = [space for space in spaces if space]
        spaces = set([sp for space in spaces for sp in space.flatSpaces])
        intersects = selfSpaces.intersection(spaces)
        if covers == 0:
            return intersects == spaces
        if covers == 1:
            return len(intersects) > 0
        else:
            return intersects == selfSpaces

    def coversActionSpaces(self, spaces):
        return self.__spaceCoverage(spaces, self.actionSpace, 0)

    def intersectsActionSpaces(self, spaces):
        return self.__spaceCoverage(spaces, self.actionSpace, 1)

    def isCoveredByActionSpaces(self, spaces):
        return self.__spaceCoverage(spaces, self.actionSpace, 2)

    def coversOutcomeSpaces(self, spaces):
        return self.__spaceCoverage(spaces, self.outcomeSpace, 0)

    def intersectsOutcomeSpaces(self, spaces):
        return self.__spaceCoverage(spaces, self.outcomeSpace, 1)

    def isCoveredByOutcomeSpaces(self, spaces):
        return self.__spaceCoverage(spaces, self.outcomeSpace, 2)

    def coversContextSpaces(self, spaces):
        return self.__spaceCoverage(spaces, self.contextSpace, 0)

    def intersectsContextSpaces(self, spaces):
        return self.__spaceCoverage(spaces, self.contextSpace, 1)

    def isCoveredByContextSpaces(self, spaces):
        return self.__spaceCoverage(spaces, self.contextSpace, 2)

    def currentContext(self, env):
        return env.observe(spaces=self.contextSpace).convertTo(self.dataset)

    def controllableContext(self):
        return self.dataset.controllableSpaces(self.contextSpace, merge=True)

    def nonControllableContext(self):
        return self.dataset.nonControllableSpaces(self.contextSpace, merge=True)
    
    # Visual

    '''def _draw_graph(self):
        """
        Draw Graph
        """
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                         node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)'''

    # Visual
    # def get_action_outcomes_visualizer(self, prefix="", context=False):
    #     """Return a dictionary used to visualize outcomes reached for the specified outcome space."""
    #     def onclick(event):
    #         center = (event['event'].xdata, event['event'].ydata)
    #         y0 = self.estimate_goal(center)[0]
    #         competence, std, dist = self.get_competence_std(center)
    #         print("{} -> {} {}".format(center, y0, competence))

    #     '''ids = []
    #     datao = self.outcomeSpace.getData()
    #     for d, id_ in zip(datao, self.outcomeSpace.ids):
    #         if d[0] ** 2 + d[1] ** 2 < 0.1:
    #             ids.append(id_)
    #     idsc = list(ids)
    #     for id_ in idsc:
    #         d = self.actionSpace.getPlainPoint([id_])
    #         if d.tolist():
    #             d = d[0]
    #             if d[0] ** 2 + d[1] ** 2 < 0.1:
    #                 ids.remove(id_)
    #         else:
    #             ids.remove(id_)
    #     data = self.actionSpace.getPlainPoint(ids)
    #     print(ids)
    #     return getVisual(
    #                     [lambda fig, ax, options: plotData(data, fig, ax, options)],
    #                     minimum=self.actionSpace.bounds['min'],
    #                     maximum=self.actionSpace.bounds['max'],
    #                     title=prefix + "Points competence for " + str(self)
    #                     )'''

    #     spaces = [self.actionSpace, self.outcomeSpace]
    #     if context:
    #         spaces.append(self.contextSpace)
    #     plots = [space.getPointsVisualizer(prefix) for space in spaces]
    #     plots[1]['onclick'] = onclick
    #     return plots

    # def get_competence_visualizer(self, dataset=None, prefix=""):
    #     data = np.array(dataset) if dataset else self.outcomeSpace.getData()
    #     print("Len {}".format(len(data)))
    #     competences = np.array([self.get_competence_std(d)[0] for d in data])
    #     ids = np.squeeze(np.argwhere(competences >= 0))
    #     competences = competences[ids]
    #     data = data[ids]
    #     print("Mean {}".format(np.mean(competences)))

    #     def onclick(event):
    #         center = (event['event'].xdata, event['event'].ydata)
    #         y0 = self.estimate_goal(center)[0]
    #         competence, std, dist = self.get_competence_std(center)
    #         print("{} -> {} {}".format(center, y0, competence))
    #     #competences = (competences - np.min(competences)) / (np.max(competences) - np.min(competences))
    #     return getVisual(
    #         [lambda fig, ax, options: plotData(
    #             data, fig, ax, options, color=competences)],
    #         minimum=self.outcomeSpace.bounds['min'],
    #         maximum=self.outcomeSpace.bounds['max'],
    #         title=prefix + "Points competence for " + str(self),
    #         colorbar=True,
    #         onclick=onclick
    #     )

    # # Plot
    # def plot(self):
    #     visualize(self.get_action_outcomes_visualizer())

    # # Api
    # def apiget(self, range_=(-1, -1)):
    #     return {'spaces': self.spacesHistory.get_range(range_), 'interest': self.interestMap.apiget(range_)}
