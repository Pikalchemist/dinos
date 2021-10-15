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
from dino.utils.maths import multivariateRegression, multivariateRegressionError, linearValue
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
    OUTLIER_MAX_DISTANCE = 100
    ENFORCE_MAX_NN_DISTANCE = False
    MAX_ERROR_REACHABLE = 0.002

    def __init__(self, dataset, actionSpace, outcomeSpace, contextSpace=[], restrictionIds=None, model=None,
                 register=True, metaData={}):
        super().__init__(dataset, actionSpace, outcomeSpace,
                         contextSpace, restrictionIds, model, register, metaData)
        self.bestLocalityCandidates = [[], []]
        self.lastCloseIds = None

    # def computeCompetence(self, error, distanceGoal=0):
    #     distanceGoal = min(distanceGoal, 1.)
    #     error = min(error, 0.1)
    #     return max(0, min(1., (1. - distanceGoal - error ** 2) / np.exp((error * 20) ** 3)))

    def reachable(self, goal: Goal, context: Observation = None, precision=0.25, precisionOrientation=0.1, inverse=True, contextColumns=None, dontMove=[], adaptContext=False):
        assert(goal is not None)
        try:
            # norm = goal.norm() / goal.space.maxDistance
            # if norm < precision * 2:
            #     precision = max(precision / 2, norm / 1.5)
            # TODO
            if inverse:
                a0, y0, _, _, _, distance, _, error = self.bestLocality(
                    goal, context, contextColumns=contextColumns, dontMove=dontMove, adaptContext=adaptContext)
            else:
                raise Exception('To check')
                # y0, goalDistanceNormalized = self.bestLocality(
                #     goal, context, getClosestOutcome=True, contextColumns=contextColumns)
                # a0 = None

            if goal.norm() < 0.001:
                a0 = a0.space.zero().setRelative(True)
                y0 = y0.space.zero().setRelative(True)
                distance = 0
                reachable = True
            if error > self.MAX_ERROR_REACHABLE:
                reachable = False

                # self.dataset.logger.info(
                #     f'{reachable}: {goal} and got {y0} ({a0}->) due to error {error} > {self.MAX_ERROR_REACHABLE}')
            else:
                if y0.norm() < 0.001:
                    orientation = 1.
                else:
                    orientation = 1. - goal.npPlain().dot(y0.npPlain()) / (goal.norm() * y0.norm())

                distance = (goal - y0).norm()
                distanceGoalNormalized = max(0, distance - goal.space.maxDistance * 0.01) / goal.norm()
                # reachable = (distanceGoalNormalized < precision and distanceGoalNormalized < 0.7) or distanceGoalNormalized < precision * 0.5
                # self.dataset.logger.info(f'HEY=== {reachable}: {goal} and got {y0} ({a0}->) distance {orientation} {distanceGoalNormalized}')

                ratio = self.outcomeSpace.number / 500
                multiplier = linearValue(1.5, 1, ratio)
                reachable = orientation < precisionOrientation and distanceGoalNormalized < precision * multiplier

                # if not reachable:
                #     self.dataset.logger.info(
                #         f'{reachable}: {goal} and got {y0} ({a0}->) distance {orientation} {distanceGoalNormalized}')
            
            # if error > self.MAX_ERROR_REACHABLE:
            #     reachable = False

            #     self.dataset.logger.info(
            #         f'{reachable}: {goal} and got {y0} ({a0}->) due to error {error} > {self.MAX_ERROR_REACHABLE}')

            return reachable, a0, y0, distance
        except ActionNotFound:
            return False, None, None, None

    def inverse(self, goal: Goal, context: Observation = None, adaptContext=False, contextColumns=None, entity=None):
        assert(goal is not None)
        a0, y0, c0, comp, error, distance = self.bestLocality(
            goal, context, adaptContext=adaptContext, contextColumns=contextColumns, entity=entity)[:6]
        return a0, y0, c0, comp, error, distance  # , dist, distNormalized

    def npForward(self, action: Action, context: Observation = None, bestContext=False, contextColumns=None,
                  ignoreFirst=False, entity=None, debug=False):
        assert(action is not None)

        results = self._nearestData(
            action, context, self.NN_LOCALITY, bestContext, outcome=False, contextColumns=contextColumns,
            nearestUseContext=True, ignoreFirst=ignoreFirst, entity=entity)
        ids, _, context, _, space, action, _, _, actionContextPlain, contextColumns = results

        # print(f'{self.actionContextSpace.getNpPlainPoint(ids)} > {self.outcomeSpace.getNpPlainPoint(ids)}')

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
        fullColumns = self.multiContextColumns(contextColumns, space, context)
        weights = space.findWeights(fullColumns, self.contextSpace)
        # c = self.contextSpace.getNpPlainPoint(ids)
        # print(f'{x} {y} {c}')

        # print(contextColumns)
        RegressionModel._db_data = (x, y, actionContextPlain, fullColumns, weights)
        y0, error = multivariateRegressionError(x, y, actionContextPlain, columns=fullColumns, weights=weights)
        maxOutliers = self.outcomeSpace.maxDistance * self.OUTLIER_MAX_DISTANCE
        if np.any(y0 > maxOutliers):
            # print('======EROEOREOROEORROEO======')
            # print(f'!!! {y0} {self.outcomeSpace.maxDistance}')
            y0 = y[-1]
            error = min(error, 0.2)

        return y0, error

    # def adaptContext(self, goal, context=None, relative=True, contextColumns=None, entity=None):
    #     if not context or not self.hasContext(self.contextSpace, contextColumns):
    #         return None

    #     self.outcomeSpace._validate()
    #     self.actionSpace._validate()

    #     goalPlain = Data.npPlainData(goal, self.outcomeSpace)

    #     # Compute best locality candidates
    #     ids, dist = self.outcomeSpace.nearest(goalPlain,
    #                                           n=self.NN_LOCALITY,
    #                                           restrictionIds=self.restrictionIds,
    #                                           otherSpace=self.actionContextSpace)

    #     # # Check if distance to goal is not too important
    #     # minPointsY = 5
    #     # idsPart = np.squeeze(np.argwhere(dist < self.outcomeSpace.maxNNDistance), axis=1)
    #     # if len(idsPart) < minPointsY:
    #     #     idsPart = range(0, len(idsPart))
    #     # ids = ids[idsPart]
    #     #
    #     # if len(ids) == 0:
    #     #     raise ActionNotFound("Not enough points to compute")

    #     print('=================================')
    #     print(dist)
    #     print(ids)
    #     try:
    #         scores = [self.inverse(goal, self.contextSpace.getPoint(id_)[0], contextColumns=contextColumns)[
    #             4] + d / 2 for id_, d in zip(ids, dist)]
    #     except ActionNotFound:
    #         scores = []
    #         for id_, d in zip(ids, dist):
    #             try:
    #                 scores.append(self.inverse(
    #                     goal, self.contextSpace.getPoint(id_)[0], contextColumns=contextColumns)[4] + d / 2)
    #             except ActionNotFound:
    #                 scores.append(-1000)
    #     ids = ids[np.argsort(scores)]
    #     print(scores)
    #     print(ids)

    #     # index
    #     index = ids[0]
    #     c0controllable = self.contextSpace.getPoint(index)[0]

    #     print(c0controllable)
    #     print(self.outcomeSpace.getPoint(index)[0])

    #     # merge
    #     nonControllable = context.projection(self.nonControllableContext(), entity=entity)
    #     c0 = c0controllable.extends(nonControllable)

    #     if relative:
    #         c0 = c0 - context
    #         c0 = c0.convertTo(kind=SpaceKind.BASIC)

    #     return c0

    def nearestOutcome(self, goal, context=None, n=1, bestContext=False, adaptContext=False, nearestUseContext=True, contextColumns=None):
        return self._nearestData(goal, context, n, bestContext, adaptContext, outcome=True, nearestUseContext=nearestUseContext,
            contextColumns=contextColumns)[:3]

    def _nearestData(self, goal, context=None, n=1, bestContext=False, adaptContext=False, outcome=True,
                     nearestUseContext=True, contextColumns=None, ignoreFirst=False, entity=None):
        self.outcomeSpace._validate()
        self.actionSpace._validate()
        if self.contextSpace:
            self.contextSpace._validate()
            self.outcomeContextSpace._validate()
            self.actionContextSpace._validate()

        if not outcome:
            contextColumns = self.contextColumns(contextColumns, goal, context)
        if not self.hasContext(self.contextSpace, contextColumns):
            context = None
        # if not outcome:
        #     print(context is None)
        #     print(contextColumns)
        if context:
            self.contextSpace._validate()
            # if adaptContext:
            #     context = self.adaptContext(goal, context=context)

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
            self.contextSpace, entity=entity) if context else None
        goalContext = goal.extends(context)
        goalContextPlain = Data.npPlainData(goalContext, space)

        # Ids
        restrictionIds = self.restrictionIds
        restrictionIdsWithZeros = self.restrictionIds
        differentRestrictionIds = False

        # Remove zero values @TODO: discrete?
        # if outcome:
        #     restrictionIds = np.unique(np.nonzero(
        #         self.outcomeSpace.getData(restrictionIds))[0])
        
        # Remove zero values @TODO: discrete?
        minNorm = self.outcomeSpace.maxDistance * 0.001
        if outcome and goal.norm() > minNorm * 2:
            differentRestrictionIds = True
            restrictionIds = self.outcomeSpace.ids[np.argwhere(np.sum(np.abs(self.outcomeSpace.getData(restrictionIds)), axis=1) > minNorm)[:, 0]]

        if context:
            contextPlain = Data.npPlainData(context, self.contextSpace)
            numberContext = int(self.NN_CONTEXT + (len(restrictionIds) if restrictionIds is not None else self.contextSpace.number) * 0.04)
            # if bestContext:
            #     ids, _ = goalSpace.nearestPlain(goalPlain, n=numberContext,
            #                                restrictionIds=restrictionIds, otherSpace=otherSpace)
            #     cids, cdists = self.contextSpace.nearestDistancePlain(contextPlain, n=self.NN_LOCALITY,
            #                                                           restrictionIds=ids, otherSpace=otherSpace, columns=contextColumns)

            #     cdistMean = np.mean(cdists[:self.NN_LOCALITY // 2])

            #     if cdistMean >= self.contextSpace.maxDistance * 0.1:
            #         # print('Trying to find a better context')
            #         context = self.contextSpace.getPoint(cids)[0]
            #         goalContext = goal.extends(context)
            #         goalContextPlain = Data.npPlainData(goalContext, space)
            #         contextPlain = Data.npPlainData(context, self.contextSpace)
            # print(contextPlain)

            restrictionIds, _ = space.nearestDistancePlain(goalContextPlain,
                                                           n=numberContext,
                                                           restrictionIds=restrictionIds,
                                                           otherSpace=otherSpace,
                                                           columns=self.multiContextColumns(contextColumns, space, context),
                                                           adjustSpaceWeight=self.contextSpace,
                                                           adjustWeightFactor=10.)
            
            # restrictionIds, _ = self.contextSpace.nearestDistancePlain(contextPlain, n=numberContext,
            #                                                            restrictionIds=restrictionIds, otherSpace=otherSpace,
            #                                                            columns=contextColumns)

            if differentRestrictionIds:
                restrictionIdsWithZeros, _ = space.nearestDistancePlain(goalContextPlain,
                                                                        n=numberContext,
                                                                        restrictionIds=restrictionIdsWithZeros,
                                                                        otherSpace=otherSpace,
                                                                        columns=self.multiContextColumns(contextColumns, space, context),
                                                                        adjustSpaceWeight=self.contextSpace,
                                                                        adjustWeightFactor=10.)
            # print('===')
            # for id_ in restrictionIds:
            #     print(f'{self.actionSpace.getPoint(id_)[0]} + {self.contextSpace.getPoint(id_)[0]} -> {self.outcomeSpace.getPoint(id_)[0]}')
        
        # print(f'HEYHEY {goalContextPlain} {self.actionContextSpace.getNpPlainPoint(restrictionIds)}')

        # print('===')
        # print(D)
        # Compute best locality candidates
        # nearestUseContext = False
        useRestrictionIds = restrictionIds
        # useRestrictionIds = self.restrictionIds
        for i in range(2):
            if nearestUseContext:
                tids, tdist = space.nearestPlain(goalContextPlain,
                                            n=n+ignoreFirst,
                                            restrictionIds=useRestrictionIds,
                                            otherSpace=otherSpace,
                                            columns=self.multiContextColumns(contextColumns, space, context),
                                            adjustSpaceWeight=self.contextSpace)
            else:
                tids, tdist = goalSpace.nearestPlain(goalPlain,
                                                n=n+ignoreFirst,
                                                restrictionIds=useRestrictionIds,
                                                otherSpace=otherSpace)
            
            if ignoreFirst and len(tids) > 0:
                tids = tids[1:]
                tdist = tdist[1:]
            
            if i == 0:
                ids = tids
                dist = tdist
            else:
                ids = np.hstack([ids, tids])
                dist = np.hstack([dist, tdist])

            if context is None or useRestrictionIds is None:
                break
            # break
            # data = goalSpace.getData(ids)
            # distanceToGoal = np.mean(np.sum(np.abs(data - goalPlain), axis=1))
            break
            useRestrictionIds = self.restrictionIds
            # nearestUseContext = False
        
        # print('===')
        # # # print(distanceToGoal)
        # print(ids)
        # print(self.actionSpace.getData(ids))
        # print(self.contextSpace.getData(ids)[:, contextColumns])
        # print(self.outcomeSpace.getData(ids))
        # RegressionModel._db_ids = (ids, contextColumns)

        # print(ids)
        # print(dist)
        # print("Distances! 3 {}".format(
        #     [self.outcomeSpace.getPoint(id_)[0] for id_ in ids]))
        # print('---')
        # for id_ in ids:
        #     print(f'{id_} {self.actionSpace.getPoint(id_)[0]} + {self.contextSpace.getPoint(id_)[0]} -> {self.outcomeSpace.getPoint(id_)[0]}')
            # print(f'{self.actionSpace.getPoint(id_)[0]} -> {self.outcomeSpace.getPoint(id_)[0]}')

        number = len(ids)
        # print(self.actionSpace.getData(ids))
        # print(self.contextSpace.getData(ids)[:, 4])
        if outcome and number > 0:
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

        # Record ids
        self.lastCloseIds = ids

        if not differentRestrictionIds:
            restrictionIdsWithZeros = restrictionIds
        return ids, dist, context, restrictionIdsWithZeros, space, goal, goalPlain, goalContext, goalContextPlain, contextColumns

    def bestLocality(self, goal: Goal, context: Observation = None, getClosestOutcome=False, bestContext=False,
                     adaptContext=False, contextColumns=None, entity=None, dontMove=[]):
        """Compute most stable local action-outcome model around goal outcome."""
        results = self._nearestData(
            goal, context, self.NN_LOCALITY*2, bestContext, adaptContext, outcome=True, contextColumns=contextColumns, entity=entity)
        ids, dist, context, restrictionIds, space, goal, goalPlain, _, goalContextPlain, contextColumns = results

        # if context:
        #     adaptContext = False

        minPointsY = 5
        minPointsA = 5
        # minNorm = self.outcomeSpace.maxDistance * 0.001

        # print(f'=> {context} {self.outcomeSpace.getPlainPoint(ids)}')

        # Check if distance to goal is not too important
        if self.ENFORCE_MAX_NN_DISTANCE:
            idsPart = np.squeeze(np.argwhere(dist < self.outcomeSpace.maxNNDistance), axis=1)
            if len(idsPart) >= minPointsY:
                ids = ids[idsPart]

        # outcomes = self.outcomeSpace.getNpPlainPoint(ids)
        # idsPart = np.squeeze(np.argwhere(np.sum(outcomes, axis=1) > minNorm), axis=1)
        # if len(idsPart) < minPointsY:
        #     idsPart = range(0, len(idsPart))
        # ids = ids[idsPart]

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

        # actionSpace = self.actionContextSpace if adaptContext else self.actionSpace
        aList = self.actionSpace.getPlainPoint(ids)
        # print(f'==> {self.outcomeSpace.getPlainPoint(ids)}')

        bestScore = None
        scores = []
        results = []

        idsAs, distAs = self.actionSpace.nearestDistanceArray(aList, n=self.NN_ALOCALITY, otherSpace=space,
                                                              restrictionIds=restrictionIds)
        for i, (idsA, distA) in enumerate(zip(idsAs, distAs)):
            # print(len(idsA))
            # print(self.outcomeSpace.getNpPlainPoint(idsA))
            # print(distA)
            # idsA, distA = self.actionSpace.nearest(p, n=10, otherSpace=space, restrictionIds=self.restrictionIds)

            # Check if distance to a is too big and enough neighbours studied already
            # idsALarge = idsA[:]
            if self.ENFORCE_MAX_NN_DISTANCE:
                idsPart = np.squeeze(np.argwhere(distA < self.actionSpace.maxNNDistance), axis=1)
                if len(idsPart) >= minPointsA:
                    idsA = idsA[idsPart]
            if len(idsA) == 0:
                continue

            yPlain = self.outcomeSpace.getNpPlainPoint(idsA)
            aPlain = self.actionSpace.getNpPlainPoint(idsA)

            # if adaptContext:
            #     cPlain = self.contextSpace.getNpPlainPoint(idsA)

            if context:
                ycPlain = self.outcomeContextSpace.getNpPlainPoint(idsA)
                # acLargePlain = self.actionContextSpace.getNpPlainPoint(idsALarge)
            else:
                ycPlain = yPlain
                # acLargePlain = aLargePlain

            distanceGoal = 0.

            columns = self.multiContextColumns(
                contextColumns, self.outcomeContextSpace, context)
            a0Plain = multivariateRegression(
                ycPlain, aPlain, goalContextPlain, columns=columns)
            maxOutliers = self.actionSpace.maxDistance * self.OUTLIER_MAX_DISTANCE
            if np.any(a0Plain > maxOutliers):
                # print(f'/!!!!\\ {a0Plain} => {aPlain[-1]}\n{aPlain}')
                a0Plain = aPlain[-1]
            a0 = self.actionSpace.action(a0Plain).setRelative(True)


            # print(f'a{i}: {a0} |a|={a0.norm()} A YC GC\n{aPlain}\n{ycPlain}\n{goalContextPlain}')

            if a0.norm() > self.actionSpace.maxDistance:
                continue

            if self.actionSpace.primitive():
                a0 = a0.bounded()

            # actionCenter = np.mean(aPlain, axis=0)
            # actionCenterDistance = np.mean(
            #     np.sum((aPlain - actionCenter) ** 2, axis=1) ** .5)
            # actionDistance = np.mean(
            #     np.sum((aPlain - a0Plain) ** 2, axis=1) ** .5)
            # proximityScore = (
            #     actionDistance / (actionCenterDistance if actionCenterDistance != 0 else 1.) - 1.) / 20.

            # if context:
            #     a0 = self.actionSpace.action(a0Plain)
            #     ac0Plain = Data.plainData(a0.extends(context), self.actionContextSpace)
            # else:
            #     ac0Plain = a0Plain
            # y0Plain, error = multivariateRegressionError(acLargePlain, yLargePlain, ac0Plain)
            y0Plain, error = self.npForward(
                a0, context, contextColumns=contextColumns)
            goalDistance = euclidean(goalPlain, y0Plain)
            # print('---')
            # print(ycPlain)
            # print(aPlain)
            # print(goalContextPlain)
            # print('>')
            # print(a0Plain)
            # print(y0Plain)
            # print(goalDistance)

            # print(f'a{i}: {a0}; error: {error}; y0/goal: {y0Plain}=?{goalPlain}; goalDistance: {goalDistance}')

            score = goalDistance / self.outcomeSpace.maxDistance
                # 0.2 * proximityScore + 0.1 * error

            result = (a0, y0Plain, goalDistance, error, i)
            if dontMove:
                scores.append(score)
                results.append(result)
            elif bestScore is None or score < bestScore:
                bestScore = score
                results = [result]
        if not results:
            raise ActionNotFound(
                "Not enough points to compute action")
        
        if dontMove:
            results = np.array(results, dtype=object)
            # print(results)
            # print(np.argsort(scores))
            result = results[np.argsort(scores)]
            move = self.dataset.checkSpaceChanges(dontMove, results[0][0], context)
            if move >= 0.1:
                scores[0] += move
                for i, result in enumerate(results):
                    if i != 0:
                        scores[i] += self.dataset.checkSpaceChanges(dontMove, result[0], context)
                result = results[np.argsort(scores)]

        a0, y0Plain, goalDistance, error, i = results[0]
        self.bestLocalityCandidates[True if dontMove else False].append(i)
        # y0Plain, goalDistance, error = bestA0Plain, bestY0Plain, bestDistance, bestError

        y0 = self.outcomeSpace.asTemplate(y0Plain, entity=entity).setRelative(True)
        if entity:
            a0 = self.actionSpace.asTemplate(a0.plain(), entity=entity).setRelative(True)
        # if a0.length() > 1000:
        #     raise Exception()
        goalDistanceNormalized = goalDistance / self.outcomeSpace.maxDistance
        finalError = (error + goalDistanceNormalized) / 2
        return (a0, y0, context, self.computeCompetence(finalError),
                finalError, goalDistance, goalDistanceNormalized, error)
