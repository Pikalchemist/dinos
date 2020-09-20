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
from exlab.interface.serializer import Serializable
from dino.data.data import *
from dino.data.space import SpaceKind
# from ..data.abstract import *
from dino.data.path import ActionNotFoundException


'''

'''


class Model(Serializable):
    number = 0
    ALMOST_ZERO_FACTOR = 0.002

    def __init__(self, dataset, actionSpace, outcomeSpace, contextSpace=[], restrictionIds=None, model=None,
                 register=True):
        self.id = Model.number
        Model.number += 1
        self.dataset = dataset

        # self.amt = AMT()
        self.actionSpace = dataset.multiColSpace(actionSpace)
        self.outcomeSpace = dataset.multiColSpace(outcomeSpace)
        self.contextSpace = dataset.multiColSpace(contextSpace)
        self.actionContextSpace = dataset.multiColSpace(
            [self.actionSpace, self.contextSpace], weight=0.5)
        self.outcomeContextSpace = dataset.multiColSpace(
            [self.outcomeSpace, self.contextSpace], weight=0.5)
        self.restrictionIds = restrictionIds

        # self.spacesHistory = DataEventHistory()

        if register:
            self.dataset.registerModel(self)

    def __repr__(self):
        return f'Model({self.actionSpace} | {self.contextSpace} => {self.outcomeSpace})'

    def _serialize(self, serializer):
        dict_ = {}
        dict_['actions'] = [s.id for s in self.actionSpace]
        dict_['outcomes'] = [s.id for s in self.outcomeSpace]
        dict_['context'] = [s.id for s in self.contextSpace]
        return dict_

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

    def update(self):
        pass

    def continueFrom(self, previousModel):
        self.id = previousModel.id
        previousModel.id = -1

        self.interestMaps = previousModel.interestMaps
        # previousModel.spacesHistory.extend(self.spacesHistory)
        self.spacesHistory = previousModel.spacesHistory

    def matches(self, model, ignoreContext=False):
        return (self.actionSpace == model.actionSpace and
                self.outcomeSpace == model.outcomeSpace and
                (ignoreContext or self.contextSpace == model.contextSpace))

    def reachesSpaces(self, spaces):
        spaces = self.dataset.convertSpaces(spaces)
        return self.outcomeSpace.intersects(spaces)

    # Space coverage
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

    def forward(self, action: Action, context: Observation = None):
        value, error = self.npForward(action, context)
        return Data(self.outcomeSpace, value.tolist()), error

    def computeCompetence(self, error, distanceGoal=0):
        distanceGoal = min(distanceGoal, 1.)
        return max(0, min(1., (1. - distanceGoal - error ** 2) / np.exp((error * 4.15) ** 3)))

    def npForward(self, action: Action, context: Observation = None):
        raise NotImplementedError()

    def inverse(self, goal: Goal, context: Observation = None):
        raise NotImplementedError()

    def bestLocality(self, goal: Goal, context: Observation = None):
        raise NotImplementedError()

    def goalCompetenceError(self, goal: Goal, context: Observation = None):
        try:
            competence, error, distance = self.inverse(goal, context)[3:6]
            return competence, error, distance
        except ActionNotFoundException:
            return -1, -1, -1

    def goalCompetence(self, goal: Goal, context: Observation = None):
        return self.goalCompetenceError(goal, context)[0]

    def getIds(self):
        ids = self.outcomeSpace.getIds(self.restrictionIds)
        ids = np.intersect1d(ids, self.actionSpace.getIds(self.restrictionIds))
        if self.contextSpace:
            ids = np.intersect1d(
                ids, self.contextSpace.getIds(self.restrictionIds))
        return ids

    def competence(self, precise=False):
        return self.computeCompetence(self.std(precise=precise))

    def eventError(self, eventId):
        action = self.actionSpace.getPoint(eventId)[0]
        outcome = self.outcomeSpace.getPoint(eventId)[0]
        context = self.contextSpace.getPoint(
            eventId)[0] if self.contextSpace else None

        actionEstimated = self.inverse(outcome, context)[0]
        actionOutcomeEstimated = self.forward(actionEstimated, context)[0]
        outcomeEstimated = self.forward(action, context)[0]

        errorAction = actionEstimated.distanceTo(
            action) / action.space.maxDistance
        errorActionOutcome = actionOutcomeEstimated.distanceTo(
            outcome) / outcome.space.maxDistance
        errorAction = min(errorAction, errorActionOutcome)
        errorOutcome = outcomeEstimated.distanceTo(
            outcome) / outcome.space.maxDistance
        # print(actionEstimated)
        # print(action)
        # print(outcomeEstimated)
        # print(outcome)
        # print(errorAction)
        # print(errorOutcome)
        # print(action.space.maxDistance)
        # print(outcome.space.maxDistance)
        return errorOutcome, errorAction

    def eventForwardError(self, eventId):
        action = self.actionSpace.getPoint(eventId)[0]
        outcome = self.outcomeSpace.getPoint(eventId)[0]
        context = self.contextSpace.getPoint(
            eventId)[0] if self.contextSpace else None

        outcomeEstimated = self.forward(action, context)[0]
        zeroError = (outcome.length() < 0.00001) != (
            outcomeEstimated.length() < 0.00001)
        errorOutcome = outcomeEstimated.distanceTo(
            outcome) / outcome.space.maxDistance + zeroError*0.0

        # if errorOutcome > 0.1:
        #     print('--- ERROR ---')
        #     print(outcome)
        #     print(outcomeEstimated)
        #     print(errorOutcome)
        #     print(context)
        #     print(action)
        #     print('---       ---')
        return errorOutcome

    def std(self, data=None, context: Observation = None, precise=False):
        # print("Variance")
        if data is None and context is None:
            errors = self._errorEvents(precise=precise)
        else:
            errors = self._errorEstimations(
                data=data, context=context, precise=precise)
        # (('Forward', errors[:, 0]), ('Inverse', errors[:, 1]))
        errorsList = (('Forward', errors),)
        # for name, errors in errorsList:
        #     print(name)
        #     print('85th percentile: {}'.format(np.percentile(errors, 85)))
        #     print('85th competence: {}'.format(self.computeCompetence(np.percentile(errors, 85))))
        #     print('median: {}'.format(np.median(errors)))
        #     print('median competence: {}'.format(self.computeCompetence(np.median(errors))))
        #     print('mean: {}'.format(np.mean(errors)))
        #     print('mean competence: {}'.format(self.computeCompetence(np.mean(errors))))
        #     print('std: {}'.format(np.std(errors)))
        #     print(np.sort(errors)[-10:])
        #     ids = np.argsort(errors)[-10:]
        #     print(ids)
        #     print()
        '''for d, c in zip(data, context):
            std = self.goalCompetenceError(d, c)[1]
            stds.append(std)'''
        # error = np.median(errors)
        errors = errors[errors < np.percentile(errors, 99)]
        # print(np.mean(errors))
        # print(np.sort(errors)[-10:])
        return np.mean(errors) + 0.08 * np.std(errors)

    def _errorEvents(self, precise=False, exceptAlmostZero=True):
        ids = self.getIds()
        if exceptAlmostZero:
            data = self.outcomeSpace.getData(ids)
            indices = np.sum(
                np.abs(data), axis=1) > self.outcomeSpace.maxDistance * self.ALMOST_ZERO_FACTOR
            if np.sum(indices) > 0:
                data = data[indices]
        if not precise:
            number = 100
            ids_ = np.arange(len(ids))
            np.random.shuffle(ids_)
            ids = ids[ids_[:number]]
        errors = np.array([self.eventForwardError(id_) for id_ in ids])
        # print(errors)
        # if not precise:
        #     print(self.outcomeSpace.getIds(self.restrictionIds)[ids_[np.argsort(errors)[-50:]]])
        # else:
        #     print(self.outcomeSpace.getIds(self.restrictionIds)[np.argsort(errors)[-50:]])
        # print(np.sort(errors)[-50:])
        return errors
        # return np.mean(errors, axis=1)

    def _errorEstimations(self, data=None, context: Observation = None, precise=False, exceptAlmostZero=True):
        data = data if data else self.outcomeSpace.getData(self.restrictionIds)
        if self.contextSpace:
            context = context if context else self.contextSpace.getData(
                self.restrictionIds)
        else:
            context = None
        if exceptAlmostZero:
            indices = np.sum(
                np.abs(data), axis=1) > self.outcomeSpace.maxDistance * self.ALMOST_ZERO_FACTOR
            if np.sum(indices) > 0:
                data = data[indices]
                if context is not None:
                    context = context[indices]
        if not precise:
            number = 100
            ids = np.arange(len(data))
            np.random.shuffle(ids)
            data = data[ids[:number]]
            if context is not None:
                context = context[ids[:number]]

        if context is not None:
            errors = [self.goalCompetenceError(self.outcomeSpace.point(d.tolist(
            )), self.contextSpace.point(c.tolist()))[1] for d, c in zip(data, context)]
        else:
            errors = [self.goalCompetenceError(self.outcomeSpace.point(d.tolist()))[
                1] for d in data]

        return errors

    def variance(self, data=None, context: Observation = None, precise=False):
        return self.std(data=data, context=context, precise=precise) ** 2

    def domainVariance(self, data=None, precise=False):
        data = data if data else self.outcomeSpace.getData(self.restrictionIds)
        if not precise:
            number = 100
            ids = np.arange(len(data))
            np.random.shuffle(ids)
            data = data[ids[:number]]

        return np.std(data)
    
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
