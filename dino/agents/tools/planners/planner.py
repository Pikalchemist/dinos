'''
    File name: planner.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

# from ..utils.debug import timethis
# from ..utils.maths import uniformRowSampling
from dino.data.data import Data, Goal, SingleAction, Action, ActionList
from dino.data.path import ActionNotFoundException, Path, Paths, PathNode
# from dino.models.model import Model
from exlab.utils.io import parameter

import numpy as np
from scipy.spatial.distance import euclidean

import itertools
import random
import math
import copy


class PlannerModelMap(object):
    def __init__(self, model):
        self.model = model
        self.iteration = -1
        self.tree = None

    def get(self):
        if self.model.dataset.getIteration() != self.iteration:
            self.iteration = self.model.dataset.getIteration()


class PlanSettings(object):
    def __init__(self, compute_actions=True):
        self.compute_actions = compute_actions
        self.length = 0
        self.depth = -1
        self.controlledSpaces = []
        self.allowContextPlanning = True
        self.mayUseContext = False

    def freeSpace(self, space):
        for s in self.controlledSpaces:
            if s.intersects(space):
                return False
        return True

    def clone(self):
        obj = copy.copy(self)
        obj.controlledSpaces = list(self.controlledSpaces)
        return obj


class Planner(object):
    """
    Plans action sequences to reach given goals.
    This Planner can use both:
    - model hierarchy: to find primitive actions correponding to a goal
    - same model skill chaining: to reach out of range goals in a given model
    """

    def __init__(self, agent, hierarchical=None, chaining=None, options={}):
        self.agent = agent
        self.environment = self.agent.environment
        self.dataset = self.agent.dataset

        self.options = options
        self.chaining = parameter(chaining, options.get('chaining', True))
        self.hierarchical = parameter(hierarchical, options.get('hierarchical', True))

        self.trees = {}
        self.semmap = None

    def plan(self, goal, hierarchical=None, state=None, model=None, settings=PlanSettings()):
        settings = settings.clone()
        return self.planDistance(goal, hierarchical=hierarchical, state=state, model=model, settings=settings)[0]

    def planDistance(self, goal, hierarchical=None, state=None, model=None, settings=PlanSettings()):
        settings = settings.clone()
        settings.depth += 1

        goal = goal.convertTo(self.dataset)
        if goal.space.primitive():
            return Paths(), 0

        if model:
            p, distance = self.__plan(
                model, goal, hierarchical=hierarchical, state=state, settings=settings)
            return Paths([p]), distance

        partition = self.partitions(goal)
        distance = 0

        paths = []
        for part in partition:
            path, dist = self.__plan(
                part[0], part[1], hierarchical=hierarchical, state=state, settings=settings)
            distance += dist
            paths.append(path)

        return Paths(paths), distance

    def planActions(self, actionList, hierarchical=None, state=None, model=None, settings=PlanSettings()):
        settings = settings.clone()
        # print('---- planActions')
        actionList = actionList.convertTo(self.dataset)
        nodes = []
        for action in actionList:
            node = PathNode(action=action)
            # print(action)
            # print(action.space.primitive())
            if not action.space.primitive():
                node.paths = self.plan(
                    action, hierarchical=hierarchical, state=state, model=model, settings=settings)
            # print(node.paths)
            nodes.append(node)
        # print('----')
        return Paths([Path(nodes)])

    def __plan(self, model, goal, hierarchical=None, state=None, settings=PlanSettings()):
        settings = settings.clone()

        # raise Exception()

        print('============= Plan =============')
        print(goal, model)

        if not settings.freeSpace(goal.space):
            print("NOT FREE!")
            print(settings.controlledSpaces)
            raise ActionNotFoundException(
                'Failed to create a path to reach {}. Space {} trying to be controlled twice!'.format(goal, goal.space), None)
        settings.controlledSpaces += model.outcomeSpace
        print(settings.controlledSpaces)

        hierarchical = hierarchical if hierarchical is not None else self.hierarchical
        space = model.outcomeSpace
        if self.environment:
            state = state if state else self.environment.state(self.dataset)
            goal = goal.relativeData(state)
        goal = goal.projection(space)

        # parameters
        range_ = 200
        goalSampleRate = 50
        minrand = -200
        maxrand = 200
        maxdist = 2 + 0.02 * goal.norm()
        lastmaxdist = 10 + 0.1 * goal.norm()
        maxiter = 30

        # init variables
        mindist = math.inf
        minnode = None
        zero = space.zero()
        zeroPlain = zero.plain()
        nodes = [PathNode(pos=zero, state=state)]
        nodePos = np.zeros((maxiter + 1, space.dim))
        path = []
        dist = 0
        directGoal = True
        # previousMove = None
        # print('GOAL: {}'.format(goal))
        # print('================')
        lastPos = None
        for i in range(maxiter):
            if random.randint(0, 100) <= goalSampleRate or directGoal:
                subgoaldistant = goal
            else:
                subgoaldistant = space.goal(
                    [random.uniform(minrand, maxrand) for x in range(space.dim)])

            # dlist = [euclidean(node.pos.plain(), subgoalPlain) for node in nodes]
            # nearestNode = nodes[dlist.index(min(dlist))]

            # print('---')
            # print(subgoal)
            # print(nearestNode.pos)
            # if move == previousMove and len(nodes) > 1:
            #     # index = np.random.randint(1, min(4, len(nodes)))
            #     # print(index)
            #     # nearestNode = nodes[np.argsort(dlist)[index]]
            #     # subgoalPlain = nodes[-1]
            #     # move = subgoal - nearestNode.pos
            #     print('same')
            # else:
            #     previousMove = move

            context = state.context() if state else None
            if self.semmap:
                context = self.semmap.context(context)

            subgoal = subgoaldistant

            subgoalPlain = subgoal.plain()
            dlist = np.sum(
                (nodePos[:len(nodes)] - subgoalPlain) ** 2, axis=1) ** 0.5
            nearestNode = nodes[np.argmin(dlist)]

            distanceToNode = 0.
            lastVariance = -1.
            # We are looking for a valid point in the outcome space
            for i in range(10):
                if i == 0:
                    move = subgoal - nearestNode.pos
                else:
                    # subgoal = subgoaldistant
                    move *= distanceToNode / move.norm()

                # moveContext = move.extends(context)
                ids, dists, c0 = model.nearestOutcome(
                    move, context=context, n=5)
                # ids, dists = model.outcomeContextSpace.nearestDistance(moveContext, n=10)
                nearestMove = model.outcomeSpace.getPoint(ids)[0]
                # print('----', i)
                # print(move)
                # print(nearestMove)
                # print(model.outcomeSpace.variance(ids))
                # print(model.outcomeSpace.denseEnough(ids))
                # print(model.outcomeSpace.maxDistance * 0.1)

                distanceToNode = 0.9 * \
                    np.mean(
                        np.sum(np.array(model.outcomeSpace.getPlainPoint(ids)) ** 2, axis=1) ** .5)
                distanceToMove = np.mean(np.sum(np.array(
                    model.outcomeSpace.getPlainPoint(ids) - move.npPlain()) ** 2, axis=1) ** .5)
                # print('!', distanceToMove)
                # print(model.outcomeSpace.getPlainPoint(ids) - move.npPlain())
                # if i == 0:
                #     distanceToNode = 10.

                variance = model.outcomeSpace.variance(ids)
                if abs(lastVariance - variance) <= 0.05 * variance:
                    break
                lastVariance = variance

                # model.outcomeSpace.denseEnough(ids, threshold=0.1) or
                if distanceToMove <= model.outcomeSpace.maxDistance * 0.25:
                    break

            # print('-->>> ', nearestMove)
            if c0:
                context = c0

            # Use whether nearest move, or directly the chosen move according to the nearest move distance
            # print('===')
            # moveDistance = (move - nearestMove).length()
            # limitDistance = model.outcomeSpace.maxDistance / 1
            # print(moveDistance)
            # if moveDistance < limitDistance and move.length() < limitDistance:
            #     # print('use')
            #     # print(moveDistance)
            #     # print(move.length())
            #     # print(nearestMove)
            #     tryDirectMove = True
            #     attemptMove = move
            #     # print('Direct?')
            # else:
            #     tryDirectMove = False
            #     attemptMove = nearestMove

            tryDirectMove = True
            attemptMove = move

            reachable, y0, a0 = model.reachable(attemptMove, context=context)
            # print('>', reachable, attemptMove, y0, a0)
            model.npForward(a0, context, debug=True)
            # print(move)
            # print(nearestMove)
            # print(y0)
            # print(euclidean((nearestNode.pos + y0).plain(), goal.plain()))
            # print(reachable)
            if not reachable:
                # print('invalid')
                if tryDirectMove:
                    reachable, y0, a0 = model.reachable(
                        nearestMove, context=context)
                    # print('REACHAAAAABLE?', reachable, y0, a0)
                    # print('>>', reachable, nearestMove, y0, a0)

                if not reachable:
                    nearestMoveHalf = nearestMove * 0.5
                    reachable, y0, a0 = model.reachable(
                        nearestMoveHalf, context=context)
                    # print('REACHAAAAABLE?', reachable, y0, a0)
                    # print('>>>', reachable, nearestMoveHalf, y0, a0)

                    if not reachable:
                        directGoal = False
                        continue
            # print('valid')
            # print(a0)
            # print(y0)

            newPos = nearestNode.pos + y0
            dist = euclidean(newPos.plain(), goal.plain())
            # print('dist', dist)
            if dist > mindist:
                # print('BOEZJPOIDVJPDVS *****')
                reachable, y0b, a0b = model.reachable(
                    nearestMove, context=context)
                # print('>>', reachable, nearestMove, y0)
                if reachable:
                    y0 = y0b
                    a0b = a0b

            newPos = nearestNode.pos + y0
            if lastPos and np.sum(np.abs(newPos.npPlain() - lastPos)) < 1.:
                break
            lastPos = newPos.plain()
            newState = nearestNode.state.copy().apply(a0, self.dataset) if state else None
            newNode = PathNode(pos=newPos, action=a0, goal=y0, model=model, parent=nearestNode,
                               state=newState)

            nodePos[len(nodes)] = newPos.npPlain()
            nodes.append(newNode)

            dist = euclidean(newPos.plain(), goal.plain())
            if dist > mindist:
                directGoal = False
            if dist < mindist:
                mindist = dist
                minnode = newNode
            # print(dist)
            if dist < maxdist:
                # print('FOUND')
                node = newNode
                while node.parent is not None:
                    path.append(node)
                    node.valid = True
                    #dist += euclidean(zeroPlain, node.goal.plain())
                    node = node.parent
                break

        if mindist < lastmaxdist and not path:
            node = minnode
            while node.parent is not None:
                path.append(node)
                node.valid = True
                node = node.parent

        path = list(reversed(path))
        # print(goal)
        # self._plotNodes(nodes, goal)
        # if path:
        #     print(path)
        '''if not path:
            goalPlain = goal.plain()
            dlist = [euclidean(node.pos.plain(), goalPlain) for node in nodes]
            nearestNode = nodes[dlist.index(min(dlist))]
            self._plotNodes(nodes, goal)
            print('Closest point {}'.format(nearestNode.pos))'''
        '''print(len(nodes))
        for node in path:
            print("x {}  (@ {})".format(node.goal, node))'''
        self._plotNodes(nodes, goal)
        if not path:
            # print('Failed!')
            p1 = None
            if settings.allowContextPlanning:
                # print('Trying context planning')
                c0, p0, d0 = self._contextPlanning(
                    model, goal, hierarchical=hierarchical, state=state, settings=settings)
                if p0:
                    lastState = p0[0][-1].state
                    # print('Successful context planning')
                    settings.controlledSpaces = settings.controlledSpaces[:-1]

                    newSettings = settings.clone()
                    newSettings.allowContextPlanning = False
                    p1, d1 = self.planDistance(
                        goal, hierarchical=hierarchical, state=lastState, settings=newSettings)
                    # print('RESULT?')
                    # print(p1)
                    if p1:
                        mindist = d1
                        paths = p0.extends(p1)
                        path = paths[0].nodes()
                        # print(path)
                        # print('^^^^^^^^')
                # print(mindist)
                # print(model)
            if not p1:
                raise ActionNotFoundException('Failed to create a path to reach {}'.format(
                    goal), mindist if mindist < math.inf else None)
        # else:
        #     print('Yes!')

        if hierarchical:
            for node in path:
                if not node.action.space.primitive():
                    # print('NODDE?')
                    node.paths = self.plan(
                        node.action, hierarchical=hierarchical, state=state)
        # print('Miaou==')

        # if settings.depth == 0:
        #     print('OVEERRRRR')
        #     print(path)

        return Path(path), max(0, mindist)

    def _contextPlanning(self, model, goal, hierarchical=None, state=None, settings=PlanSettings()):
        settings = settings.clone()
        path = []
        print(model.controllableContext())
        if not model.controllableContext():
            return None, None, None

        c0 = model.adaptContext(goal, state.context())
        if not c0:
            return None, None, None

        try:
            cpath, cdistance = self.planDistance(
                c0, hierarchical=hierarchical, state=state, settings=settings)
        except Exception:
            return None, None, None
        print('--------')
        print(cpath[0][-1].state)
        print(c0)

        return c0, cpath, cdistance

        # model.inverse()
