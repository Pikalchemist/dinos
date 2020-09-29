import sys
import copy
import math
import random
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean

# import matplotlib as plt

from .model import Model

# from ..utils.logging import DataEventHistory, DataEventKind
# from ..utils.io import getVisual, plotData, visualize, concat
# from ..utils.serializer import serialize
# from ..utils.maths import multivariateRegression, multivariateRegressionError
from dino.utils.maths import multivariateRegression, multivariateRegressionError
from dino.data.data import Data, Goal, Observation, Action
# from dino.data.abstract import *
from dino.data.space import SpaceKind
from dino.data.path import ActionNotFound


class RegressionModel(Model):
    NN_CONTEXT = 40
    NN_CONTEXT_RATIO = 0.05
    NN_ALOCALITY = 5
    NN_LOCALITY = 10
    MINIMUM_LENGTH = 0.001
    MAXIMUM_NULL = 5

    def __init__(self, dataset, actionSpace, outcomeSpace, contextSpace=[], restrictionIds=None, model=None,
                 register=True):
        super().__init__(dataset, actionSpace, outcomeSpace,
                         contextSpace, restrictionIds, model, register)

    def computeCompetence(self, error, distanceGoal=0):
        distanceGoal = min(distanceGoal, 1.)
        error = min(error, 0.1)
        return max(0, min(1., (1. - distanceGoal - error ** 2) / np.exp((error * 20) ** 3)))

    def reachable(self, goal: Goal, context: Observation = None, precision=0.05, inverse=True, contextColumns=None):
        assert(goal is not None)
        try:
            # norm = goal.norm() / goal.space.maxDistance
            # if norm < precision * 2:
            #     precision = max(precision / 2, norm / 1.5)
            # TODO
            if inverse:
                a0, y0, _, _, _, _, _, goalSpaceDistanceNormalized = self.bestLocality(
                    goal, context, contextColumns=contextColumns)
            else:
                y0, goalSpaceDistanceNormalized = self.bestLocality(
                    goal, context, getClosestOutcome=True, contextColumns=contextColumns)
                a0 = None

            if goal.norm() == 0:
                reachable = True
            else:
                if y0.norm() == 0:
                    orientation = 1.
                else:
                    orientation = 1. - goal.npPlain().dot(y0.npPlain()) / (goal.norm() * y0.norm())
                goalDistanceNormalized = (goal - y0).norm() / goal.norm()
                reachable = (goalSpaceDistanceNormalized < precision and goalDistanceNormalized < 0.7) or goalSpaceDistanceNormalized < precision * 0.5
                if orientation < precision and y0.norm() > 0.2 * goal.norm():
                    reachable = True

                # if not reachable:
                #     self.dataset.logger.info(
                #         f'{goal} and got {y0} ({a0}->) distance {orientation} {goalSpaceDistanceNormalized} {goalDistanceNormalized}')
                #     print('=== Hey ===')
                #     print(goalSpaceDistanceNormalized)
                #     print(goalDistanceNormalized)
                #     print(self.outcomeSpace.maxDistance)
                #     print(precision)
                #     print(y0)
                #     print(goal)
            return reachable, y0, a0
        except ActionNotFound:
            # print('=== Hey ===')
            # print(goal)
            return False, None, None

    def inverse(self, goal: Goal, context: Observation = None, adaptContext=False, contextColumns=None):
        assert(goal is not None)
        a0, y0, c0, comp, error, distance = self.bestLocality(
            goal, context, adaptContext=adaptContext, contextColumns=contextColumns)[:6]
        return a0, y0, c0, comp, error, distance  # , dist, distNormalized

    def npForward(self, action: Action, context: Observation = None, bestContext=True, contextColumns=None, ignoreFirst=False, debug=False):
        assert(action is not None)
        # contextColumns = self.contextColumns(contextColumns, )
        # import pylab as plt

        # action = action.projection(self.actionSpace)
        # actionPlain = Data.npPlainData(action, self.actionSpace)
        # space = self.actionContextSpace if context else self.actionSpace
        # context = context.convertTo(kind=SpaceKind.PRE).projection(self.contextSpace) if context else None
        # actionContext = action.extends(context)
        # actionContextPlain = Data.npPlainData(actionContext, space)
        #
        # restrictionIds = self.restrictionIds
        # if context:
        #     contextPlain = Data.npPlainData(context, self.contextSpace)
        #     restrictionIds, dists = self.contextSpace.nearestDistance(contextPlain, n=100, restrictionIds=restrictionIds, otherSpace=self.outcomeSpace)
        #     # print(actionContextPlain)
        #     # print(action)
        #     # print(context)
        #     # print('?')
        #     # restrictionIds, dists = self.actionContextSpace.nearestDistance(actionContextPlain,
        #     #                                                                 n=self.NN_CONTEXT,
        #     #                                                                 restrictionIds=self.restrictionIds,
        #     #                                                                 otherSpace=self.outcomeSpace)
        #     #
        #     # print(dists)
        #     # print(self.actionContextSpace.getPlainPoint(restrictionIds)[:10])
        #
        #     if debug:
        #         print("Context {}".format(contextPlain))
        #         print("Selecting 1 {}".format(restrictionIds))
        #         print("Distances 1 {}".format([action.distanceTo(self.actionSpace.getPoint(id_)[0]) for id_ in restrictionIds]))
        #         print("Distances 2 {}".format([context.distanceTo(self.contextSpace.getPoint(id_)[0]) for id_ in restrictionIds]))
        #
        # if context:
        #     ids, dists = self.actionContextSpace.nearestDistance(actionContextPlain,
        #                                                          n=self.NN_LOCALITY,
        #                                                          restrictionIds=restrictionIds,
        #                                                          otherSpace=self.outcomeSpace)
        # else:
        #     ids, dists = self.actionSpace.nearestDistance(actionPlain,
        #                                                   n=self.NN_LOCALITY,
        #                                                   restrictionIds=restrictionIds,
        #                                                   otherSpace=self.outcomeContextSpace)

        results = self._nearestData(
            action, context, self.NN_LOCALITY, bestContext, outcome=False, contextColumns=contextColumns, nearestUseContext=True, ignoreFirst=ignoreFirst)
        ids, dist, context, restrictionIds, space, action, actionPlain, actionContext, actionContextPlain = results

        # if debug:
        #     print('=========')
        #     print(action)
        #     print(np.array(self.actionSpace.getPlainPoint(ids)))
        #     print(dist)
        #     print(context)
        # print(self.contextSpace.getPlainPoint(restrictionIds))
        # print("Selecting 2 {}".format(ids))
        # print("Distances 3 {}".format(dists))

        # ids = self.actionSpace.getActionIndex(ids)
        if len(ids) == 0:
            return None, None

        x = space.getNpPlainPoint(ids)
        y = self.outcomeSpace.getNpPlainPoint(ids)
        # if debug:
        #     print(y)
        #     print(multivariateRegressionError(x, y, actionContextPlain))

        return multivariateRegressionError(x, y, actionContextPlain, columns=self.multiContextColumns(contextColumns, space))

    def adaptContext(self, goal, context=None, relative=True, contextColumns=None):
        if not context or not self.hasContext(self.contextSpace, contextColumns):
            return None

        self.outcomeSpace._validate()
        self.actionSpace._validate()

        goalPlain = Data.npPlainData(goal, self.outcomeSpace)

        # Compute best locality candidates
        ids, dist = self.outcomeSpace.nearest(goalPlain,
                                              n=self.NN_LOCALITY,
                                              restrictionIds=self.restrictionIds,
                                              otherSpace=self.actionContextSpace)

        # # Check if distance to goal is not too important
        # minPointsY = 5
        # idsPart = np.squeeze(np.argwhere(dist < self.outcomeSpace.maxNNDistance), axis=1)
        # if len(idsPart) < minPointsY:
        #     idsPart = range(0, len(idsPart))
        # ids = ids[idsPart]
        #
        # if len(ids) == 0:
        #     raise ActionNotFound("Not enough points to compute")

        print('=================================')
        print(dist)
        print(ids)
        try:
            scores = [self.inverse(goal, self.contextSpace.getPoint(id_)[0], contextColumns=contextColumns)[
                4] + d / 2 for id_, d in zip(ids, dist)]
        except ActionNotFound:
            scores = []
            for id_, d in zip(ids, dist):
                try:
                    scores.append(self.inverse(
                        goal, self.contextSpace.getPoint(id_)[0], contextColumns=contextColumns)[4] + d / 2)
                except ActionNotFound:
                    scores.append(-1000)
        ids = ids[np.argsort(scores)]
        print(scores)
        print(ids)

        # index
        index = ids[0]
        c0controllable = self.contextSpace.getPoint(index)[0]

        print(c0controllable)
        print(self.outcomeSpace.getPoint(index)[0])

        # merge
        nonControllable = context.projection(self.nonControllableContext())
        c0 = c0controllable.extends(nonControllable)

        if relative:
            c0 = c0 - context
            c0 = c0.convertTo(kind=SpaceKind.BASIC)

        return c0

    def nearestOutcome(self, goal, context=None, n=1, bestContext=True, adaptContext=False, nearestUseContext=True, contextColumns=None):
        return self._nearestData(goal, context, n, bestContext, adaptContext, outcome=True, nearestUseContext=nearestUseContext,
            contextColumns=contextColumns)[:3]

    def _nearestData(self, goal, context=None, n=1, bestContext=True, adaptContext=False, outcome=True, nearestUseContext=True, contextColumns=None, ignoreFirst=False):
        self.outcomeSpace._validate()
        self.actionSpace._validate()
        if not self.hasContext(self.contextSpace, contextColumns):
            context = None
        if context:
            self.contextSpace._validate()
            if adaptContext:
                context = self.adaptContext(goal, context=context)

        if outcome:
            goalSpace = self.outcomeSpace
            goalContextSpace = self.outcomeContextSpace
            otherSpace = self.actionSpace
        else:
            goalSpace = self.actionSpace
            goalContextSpace = self.actionContextSpace
            otherSpace = self.outcomeSpace

        goal = goal.projection(goalSpace)
        goalPlain = Data.npPlainData(goal, goalSpace)
        space = goalContextSpace if context else goalSpace
        context = context.convertTo(self.dataset, kind=SpaceKind.PRE).projection(
            self.contextSpace) if context else None
        goalContext = goal.extends(context)
        goalContextPlain = Data.plainData(goalContext, space)

        # Ids
        restrictionIds = self.restrictionIds

        # Remove zero values @TODO: discrete?
        if outcome:
            restrictionIds = np.unique(np.nonzero(
                self.outcomeSpace.getData(restrictionIds))[0])

        if context:
            contextPlain = Data.npPlainData(context, self.contextSpace)
            numberContext = int(self.NN_CONTEXT + (len(restrictionIds)
                                                   if restrictionIds is not None else self.contextSpace.number) * 0.04)
            # print('numberContext', numberContext)
            # print(contextPlain)
            if bestContext:
                ids, _ = goalSpace.nearest(goalPlain, n=numberContext,
                                           restrictionIds=restrictionIds, otherSpace=otherSpace)
                cids, cdists = self.contextSpace.nearestDistance(contextPlain, n=self.NN_LOCALITY,
                                                                 restrictionIds=ids, otherSpace=otherSpace, columns=contextColumns)

                cdistMean = np.mean(cdists[:self.NN_LOCALITY // 2])
                # print(cdists)
                # print('dists', cdistMean, self.contextSpace.maxDistance * 0.1)

                if cdistMean >= self.contextSpace.maxDistance * 0.1:
                    # print('Trying to find a better context')
                    # print(context)
                    context = self.contextSpace.getPoint(cids)[0]
                    # print(context)
                    goalContext = goal.extends(context)
                    goalContextPlain = Data.plainData(goalContext, space)
            # print(contextPlain)

            restrictionIds, _ = self.contextSpace.nearestDistance(contextPlain, n=numberContext,
                                                                  restrictionIds=restrictionIds, otherSpace=otherSpace,
                                                                  columns=contextColumns)
            # restrictionIds, distContext = self.outcomeContextSpace.nearestDistance(goalContextPlain,
            #                                                                        n=numberContext,
            #                                                                        restrictionIds=restrictionIds,
            #                                                                        otherSpace=self.actionSpace)

            # if bestContext:
            #     print('Trying to find a better context', np.mean(distContext))
            #     print(context)
            #     print()

            # Fallback if no point found
            # print(distContext)
            # if len(restrictionIds) == 0:
            #     print(f'Fallback for {goal} without context {context}')
            #     restrictionIds = self.restrictionIds

            # print("Context! {}".format(contextPlain))
            # print("Selecting! 1 {}".format(restrictionIds))
            # print("Distances! 1 {}".format([goal.distanceTo(self.outcomeSpace.getPoint(id_)[0]) for id_ in restrictionIds]))
            # print("Distances! 2 {}".format([context.distanceTo(self.contextSpace.getPoint(id_)[0]) for id_ in restrictionIds]))
            # print("Distances! 1 {}".format(
            #     [self.outcomeSpace.getPoint(id_)[0] for id_ in restrictionIds]))
            # print("Distances! 2 {}".format(
            #     [self.contextSpace.getPoint(id_)[0] for id_ in restrictionIds]))
            # print('===')
            # for id_ in restrictionIds:
            #     print(f'{self.actionSpace.getPoint(id_)[0]} + {self.contextSpace.getPoint(id_)[0]} -> {self.outcomeSpace.getPoint(id_)[0]}')

        # Compute best locality candidates
        if nearestUseContext:
            ids, dist = space.nearest(goalContextPlain,
                                      n=n+ignoreFirst,
                                      restrictionIds=restrictionIds,
                                      otherSpace=otherSpace,
                                      columns=self.multiContextColumns(contextColumns, space))
        else:
            ids, dist = goalSpace.nearest(goalPlain,
                                          n=n+ignoreFirst,
                                          restrictionIds=restrictionIds,
                                          otherSpace=otherSpace)
        if ignoreFirst and len(ids) > 0:
            ids = ids[1:]
            dist = dist[1:]
        # print(ids)
        # print(dist)
        # print("Distances! 3 {}".format(
        #     [self.outcomeSpace.getPoint(id_)[0] for id_ in ids]))
        # print('---')
        # for id_ in ids:
        #     print(f'{id_} {self.actionSpace.getPoint(id_)[0]} + {self.contextSpace.getPoint(id_)[0]} -> {self.outcomeSpace.getPoint(id_)[0]}')
            # print(f'{self.actionSpace.getPoint(id_)[0]} -> {self.outcomeSpace.getPoint(id_)[0]}')

        number = len(ids)
        if number > 0:
            data = self.outcomeSpace.getData(ids)
            mean = np.mean(data, axis=0)
            for zero in (True, False):
                if zero:
                    distanceToCenter = np.sum(np.abs(data), axis=1)
                else:
                    distanceToCenter = np.sum(np.abs(data - mean), axis=1)
                indices = distanceToCenter > self.outcomeSpace.maxDistance * self.ALMOST_ZERO_FACTOR
                if np.sum(indices) < number * 3 // 10:
                    ids = ids[~indices]
                    dist = dist[~indices]
                    data = data[~indices]
                elif np.sum(~indices) < number * 3 // 10:
                    ids = ids[indices]
                    dist = dist[indices]
                    data = data[indices]
            # if np.sum(indices) > 0:
            #     data = data[indices]

        return ids, dist, context, restrictionIds, space, goal, goalPlain, goalContext, goalContextPlain

    def bestLocality(self, goal: Goal, context: Observation = None, getClosestOutcome=False, bestContext=True, adaptContext=False, contextColumns=None):
        """Compute most stable local action-outcome model around goal outcome."""
        contextColumns = self.contextColumns(contextColumns, goal, context)
        if not self.hasContext(self.contextSpace, contextColumns):
            context = None
        # self.outcomeSpace._validate()
        # self.actionSpace._validate()
        # if context:
        #     self.contextSpace._validate()
        #     if adaptContext:
        #         context = self.adaptContext(goal, context=context)

        # goal = goal.projection(self.outcomeSpace)
        # goalPlain = Data.npPlainData(goal, self.outcomeSpace)
        # space = self.outcomeContextSpace if context else self.outcomeSpace
        # context = context.projection(self.contextSpace) if context else None
        # goalContext = goal.extends(context)
        # goalContextPlain = Data.plainData(goalContext, space)

        # print(goalContextPlain)

        # restrictionIds = self.restrictionIds

        # Remove zero values @TODO: discrete?
        # restrictionIds = np.unique(np.nonzero(self.outcomeSpace.getData(restrictionIds))[0])
        # numberNull = np.sum(np.sum(np.abs(self.outcomeSpace.getData(restrictionIds)), axis=1) < self.MINIMUM_LENGTH)

        # if context:
        #     contextPlain = Data.npPlainData(context, self.contextSpace)
        #     restrictionIds, _ = self.contextSpace.nearestDistance(contextPlain, n=self.NN_CONTEXT, restrictionIds=restrictionIds, otherSpace=self.actionSpace)
        #     # restrictionIds, _ = self.outcomeContextSpace.nearestDistance(goalContextPlain,
        #     #                                                              n=self.NN_CONTEXT,
        #     #                                                              restrictionIds=restrictionIds,
        #     #                                                              otherSpace=self.actionSpace)
        #
        #     # Fallback if no point found
        #     if len(restrictionIds) == 0:
        #         print("Fallback for {} without context {}".format(goal, context))
        #         restrictionIds = self.restrictionIds

        # print("Context! {}".format(contextPlain))
        # print("Selecting! 1 {}".format(restrictionIds))
        # print("Distances! 1 {}".format([goal.distanceTo(self.outcomeSpace.getPoint(id_)[0]) for id_ in restrictionIds]))
        # print("Distances! 2 {}".format([context.distanceTo(self.contextSpace.getPoint(id_)[0]) for id_ in restrictionIds]))

        # Compute best locality candidates
        # ids, dist = self.outcomeSpace.nearest(goalPlain,
        #                                       n=self.NN_LOCALITY,
        #                                       restrictionIds=restrictionIds,
        #                                       otherSpace=self.actionContextSpace)
        # ids, dist = self.outcomeContextSpace.nearest(goalContextPlain,
        #                                              n=self.NN_LOCALITY,
        #                                              restrictionIds=restrictionIds,
        #                                              otherSpace=self.actionSpace)

        results = self._nearestData(
            goal, context, self.NN_LOCALITY, bestContext, adaptContext, outcome=True, contextColumns=contextColumns)
        ids, dist, context, restrictionIds, space, goal, goalPlain, goalContext, goalContextPlain = results

        minPointsY = 5
        minPointsA = 5

        # Check if distance to goal is not too important
        idsPart = np.squeeze(np.argwhere(
            dist < self.outcomeSpace.maxNNDistance), axis=1)
        if len(idsPart) < minPointsY:
            idsPart = range(0, len(idsPart))
        ids = ids[idsPart]

        if len(ids) == 0:
            raise ActionNotFound("Not enough points to compute")

        if getClosestOutcome:
            distanceContext = (self.contextSpace.getPoint(restrictionIds[0])[0].distanceTo(context) /
                               self.contextSpace.maxDistance)
            y0 = self.outcomeSpace.getPoint(ids[0])[0]
            distanceGoal = goal.distanceTo(y0) / self.outcomeSpace.maxDistance
            # print(distanceContext)
            # print(distanceGoal)
            return y0, distanceGoal + distanceContext

        aList = self.actionSpace.getPlainPoint(ids)

        # print("=====")
        # print(self.outcomeSpace.getNpPlainPoint(ids))
        # print("=====")
        # print(self.contextSpace.getNpPlainPoint(restrictionIds))

        # print(dist)
        # print(self.actionSpace.getPlainPoint(ids)[:2])
        # print(self.outcomeSpace.getPlainPoint(ids)[:2])
        # print(self.contextSpace.getPlainPoint(ids)[:2])

        bestScore = -1
        minDistance = -1
        # minStd = -1
        minY0Plain = None
        minA0Plain = None

        idsAs, distAs = self.actionSpace.nearestDistanceArray(aList, n=self.NN_ALOCALITY, otherSpace=space,
                                                              restrictionIds=restrictionIds)
        if len(idsAs) > 0:
            for i, p in enumerate(aList):
                idsA, distA = idsAs[i], distAs[i]
                # print(len(idsA))
                # print(self.outcomeSpace.getNpPlainPoint(idsA))
                # print(distA)
                # idsA, distA = self.actionSpace.nearest(p, n=10, otherSpace=space, restrictionIds=self.restrictionIds)

                # Check if distance to a is too big and enough neighbours studied already
                # idsALarge = idsA[:]
                idsPart = np.squeeze(np.argwhere(
                    distA[:min(self.NN_LOCALITY, len(idsA))] < self.actionSpace.maxNNDistance), axis=1)
                if len(idsPart) < minPointsA:
                    idsPart = range(len(idsPart))
                idsA = idsA[idsPart]
                if len(idsA) == 0:
                    continue

                yPlain = self.outcomeSpace.getNpPlainPoint(idsA)
                aPlain = self.actionSpace.getNpPlainPoint(idsA)

                # print('===')
                # for id_ in idsA:
                #     print(f'{self.actionSpace.getPoint(id_)[0]} + {self.contextSpace.getPoint(id_)[0]} -> {self.outcomeSpace.getPoint(id_)[0]}')

                # yLargePlain = self.outcomeSpace.getNpPlainPoint(idsALarge)
                # aLargePlain = self.actionSpace.getNpPlainPoint(idsALarge)
                # print(idsA)
                if context:
                    ycPlain = self.outcomeContextSpace.getNpPlainPoint(idsA)
                    # acLargePlain = self.actionContextSpace.getNpPlainPoint(idsALarge)
                else:
                    ycPlain = yPlain
                    # acLargePlain = aLargePlain

                distanceGoal = 0.

                # numberNull = np.sum(np.sum(np.abs(yPlain), axis=1) < self.MINIMUM_LENGTH)
                # if numberNull >= self.MAXIMUM_NULL:
                #     # print(yPlain)
                #     # print(np.sum(yPlain, axis=1))
                #     # print(np.sum(yPlain, axis=1) < self.MINIMUM_LENGTH)
                #     # print('Not')
                #     continue

                # print('===')
                # print(yPlain)
                # print(ycPlain)
                # print(aPlain)
                # print(goalContextPlain)
                # print('===')
                # print(ycPlain)
                # print(aPlain)
                columns = self.multiContextColumns(
                    contextColumns, self.outcomeContextSpace)
                a0Plain = multivariateRegression(
                    ycPlain, aPlain, goalContextPlain, columns=columns)
                a0 = self.actionSpace.action(a0Plain)

                actionCenter = np.mean(aPlain, axis=0)
                actionCenterDistance = np.mean(
                    np.sum((aPlain - actionCenter) ** 2, axis=1) ** .5)
                actionDistance = np.mean(
                    np.sum((aPlain - a0Plain) ** 2, axis=1) ** .5)
                proximityScore = (
                    actionDistance / (actionCenterDistance if actionCenterDistance != 0 else 1.) - 1.) / 20.

                # if context:
                #     a0 = self.actionSpace.action(a0Plain)
                #     ac0Plain = Data.plainData(a0.extends(context), self.actionContextSpace)
                # else:
                #     ac0Plain = a0Plain
                # y0Plain, error = multivariateRegressionError(acLargePlain, yLargePlain, ac0Plain)
                y0Plain, error = self.npForward(
                    a0, context, contextColumns=contextColumns)
                distanceGoal = euclidean(goalPlain, y0Plain)
                # print('---')
                # print(ycPlain)
                # print(aPlain)
                # print(goalContextPlain)
                # print('>')
                # print(a0Plain)
                # print(y0Plain)
                # print(distanceGoal)

                score = distanceGoal / self.outcomeSpace.maxDistance# + \
                    # 0.2 * proximityScore + 0.1 * error

                # print('moa')
                # print(a0Plain, y0Plain)
                # print(distanceGoal)
                # print(error)
                # print(proximityScore)
                # print(score)
                # print()

                # print(f'Score {score} a {a0} y0 {y0Plain} {goalPlain}')
                if bestScore < 0 or score < bestScore:
                    bestScore = score
                    minDistance = distanceGoal
                    minError = error
                    # minYPlain = yPlain
                    # minAPlain = aPlain
                    minY0Plain = y0Plain
                    minA0Plain = a0Plain
        if minY0Plain is None:
            raise ActionNotFound(
                "Not enough points to compute action")

        y0 = self.outcomeSpace.asTemplate(minY0Plain)
        a0 = self.actionSpace.asTemplate(minA0Plain)
        # if a0.length() > 1000:
        #     raise Exception()
        goalDistanceNormalized = minDistance / self.outcomeSpace.maxDistance
        error = (minError + goalDistanceNormalized) / 2
        return (a0, y0, context, self.computeCompetence(error),
                error, minDistance, minError, goalDistanceNormalized)
