'''
    File name: planner.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

from exlab.modular.module import Module
from exlab.utils.io import parameter

# from ..utils.debug import timethis
# from ..utils.maths import uniformRowSampling
from dino.data.data import Data, Goal, SingleAction, Action, ActionList
from dino.data.path import ActionNotFound, Path, PathNode
from dino.data.space import SpaceKind
# from dino.models.model import Model

from collections import namedtuple

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
        self.dontMoveSpaces = []

        self.allowContextPlanning = True
        self.mayUseContext = False
        self.hierarchical = None

    def freeSpace(self, space):
        for s in self.controlledSpaces:
            if s.intersects(space):
                return False
        return True

    def clone(self):
        obj = copy.copy(self)
        obj.controlledSpaces = list(self.controlledSpaces)
        return obj


Part = namedtuple('Part', ['model', 'goal'])


class Planner(Module):
    """
    Plans action sequences to reach given goals.
    This Planner can use both:
    - model hierarchy: to find primitive actions correponding to a goal
    - same model skill chaining: to reach out of range goals in a given model
    """

    MAX_MOVE_UNCHANGED_SPACES = 1.

    def __init__(self, agent, hierarchical=None, chaining=None, options={}):
        super().__init__('planner', agent)
        self.logger.tag = 'plan'

        self.agent = agent
        self.environment = self.agent.environment
        self.dataset = self.agent.dataset

        self.options = options
        self.chaining = parameter(chaining, options.get('chaining', True))
        self.hierarchical = parameter(hierarchical, options.get('hierarchical', True))

        self.trees = {}
        self.semmap = None
    
    def partitions(self, goal, settings=PlanSettings()):
        settings = settings.clone()

        parts = goal.flat()
        space = goal.space

        self.logger.debug(f'Parting space to find models reaching {space}')

        for _ in range(1):
            # Find models related to our goal
            # models [(model, goal, related_parts, prob), ...]
            partsLeft = list(parts)
            partition = []
            while partsLeft:
                models = []
                for model in self.dataset.models:
                    spaces = list(model.reachesSpaces([p.space for p in partsLeft]))
                    if len(spaces) > 0 and settings.freeSpace(space):
                        models.append((model, spaces, model.competence()))

                if not models:
                    raise ActionNotFound(f'No model found planning how to reach {goal}')

                models = np.array(models)
                models = models[np.argsort(models[:, 2])]
                bestModel = models[0]
                # model = uniformRowSampling(models, [prob for _, _, prob in models])

                partsLeft = [p for p in partsLeft if p.space not in bestModel[1]]
                partition.append(
                    Part(bestModel[0], Goal(*[p for p in parts if p.space in bestModel[1]])))
        return partition

    def plan(self, goal, state=None, model=None, settings=PlanSettings()):
        return self.planDistance(goal, state=state, model=model, settings=settings)[0]

    def planDistance(self, goal, state=None, model=None, settings=PlanSettings()):
        settings = settings.clone()
        settings.depth += 1

        goal = goal.convertTo(self.dataset, kind=SpaceKind.BASIC)
        if goal.space.primitive():
            raise Exception(f'Primitive actions such as {goal} cannot be goal, thus not planned!')

        if model:
            partition = [Part(model, goal)]
        else:
            partition = self.partitions(goal)
        # Using only 1 element from the partition
        partition = [partition[0]]

        totalDistance = 0
        paths = []
        for model, subgoal in partition:
            path, dist = self.__plan(
                model, subgoal, state=state, settings=settings)
            totalDistance += dist
            paths.append(path)

        return paths[0], totalDistance

    def planActions(self, actionList, state=None, model=None, settings=PlanSettings()):
        settings = settings.clone()

        # print('---- planActions')
        actionList = actionList.convertTo(self.dataset)
        nodes = []
        for action in actionList:
            node = PathNode(action=action)
            # print(action)
            # print(action.space.primitive())
            if not action.space.primitive():
                node.execution = self.plan(
                    action, state=state, model=model, settings=settings)
            # print(node.paths)
            nodes.append(node)
        # print('----')
        return Path(nodes)

    def __plan(self, model, goal, state=None, settings=PlanSettings()):
        settings = settings.clone()

        self.logger.debug(f'=== New planning (d{settings.depth}) === -> {goal} using {model} with context {state}')

        # Checking if space already planned
        if not settings.freeSpace(goal.space):
            error = f'Failed to create a path to reach {goal}. Space {goal.space} trying to be controlled twice! (controlled spaces: {settings.controlledSpaces})'
            self.logger.debug(error)
            raise ActionNotFound(error, None)
        settings.controlledSpaces += model.outcomeSpace
        print(settings.controlledSpaces)

        # Settings
        hierarchical = parameter(settings.hierarchical, self.hierarchical)
        space = model.outcomeSpace
        if self.environment:
            state = state if state else self.environment.state(self.dataset)
            goal = goal.relativeData(state)
        goal = goal.projection(space)

        attemptedUnreachable = []
        attemptedBreakConstraint = []
        lastInvalids = []

        # parameters
        range_ = 200
        goalSampleRate = 50
        minrand = -200
        maxrand = 200
        maxdist = 2 + 0.02 * goal.norm()
        maxdistIncomplete = 5 + 0.1 * goal.norm()
        # lastmaxdist = 10 + 0.1 * goal.norm()
        maxiter = 100

        # init variables
        mindist = math.inf
        minnode = None
        zero = space.zero()
        zeroPlain = zero.plain()
        nodes = [PathNode(pos=zero, state=state)]
        nodePositions = np.zeros((maxiter + 1, space.dim))
        path = Path()
        dist = 0
        directGoal = True
        # previousMove = None
        # print('GOAL: {}'.format(goal))
        # print('================')
        lastPos = None

        invalidPointRatioLimit = 0.02
        ignoreConstraintDistanceLimit = 0.05 * space.maxDistance

        # Main loop
        for i in range(maxiter):
            # Finding a subgoal
            if random.randint(0, 100) <= goalSampleRate or directGoal:
                subgoaldistant = goal
            else:
                subgoaldistant = space.goal(
                    [random.uniform(minrand, maxrand) for x in range(space.dim)])

            if lastInvalids:
                invalids = np.array(lastInvalids[-10:])
                dists = np.sum((invalids - subgoaldistant.npPlain()) ** 2, axis=1) ** 0.5
                if np.any(dists < invalidPointRatioLimit * space.maxDistance):
                    self.logger.debug2(f'(d{settings.depth}) Iter {i}: Too close to invalid point, retrying...')


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

            self.logger.debug2(
                f'(d{settings.depth}) Iter {i}: chosen subgoal is {subgoal}(->{subgoaldistant}) (direct={directGoal}) (final goal {goal}) with context {state}')

            # Nearest Node
            subgoalPlain = subgoal.plain()
            dlist = np.sum(
                (nodePositions[:len(nodes)] - subgoalPlain) ** 2, axis=1) ** 0.5
            nearestNode = nodes[np.argmin(dlist)]

            distancesToNode = 0.
            lastVariance = -1.
            # We are looking for a valid point in the outcome space
            for j in range(10):
                if j == 0:
                    move = subgoal - nearestNode.pos
                else:
                    # subgoal = subgoaldistant
                    move *= distancesToNode / move.norm()

                # moveContext = move.extends(context)
                ids, _, _ = model.nearestOutcome(
                    move, context=context, n=5)
                # ids, dists = model.outcomeContextSpace.nearestDistance(moveContext, n=10)
                nearestMove = model.outcomeSpace.getPoint(ids)[0]
                nearestMoveId = ids[0]
                # print('----', j)
                # print(move)
                # print(nearestMove)
                # print(model.outcomeSpace.variance(ids))
                # print(model.outcomeSpace.denseEnough(ids))
                # print(model.outcomeSpace.maxDistance * 0.1)

                distancesToNode = 0.9 * np.mean(np.sum(np.array(
                    model.outcomeSpace.getPlainPoint(ids) - nearestNode.pos.npPlain()) ** 2, axis=1) ** .5)
                distancesToMove = np.mean(np.sum(np.array(
                    model.outcomeSpace.getPlainPoint(ids) - move.npPlain()) ** 2, axis=1) ** .5)
                # print('!', distancesToMove)
                # print(model.outcomeSpace.getPlainPoint(ids) - move.npPlain())
                # if j == 0:
                #     distancesToNode = 10.

                variance = model.outcomeSpace.variance(ids)
                if abs(lastVariance - variance) <= 0.05 * variance:
                    break
                lastVariance = variance

                # model.outcomeSpace.denseEnough(ids, threshold=0.1) or
                if distancesToMove <= model.outcomeSpace.maxDistance * 0.25:
                    break

            # print('-->>> ', c0)
            # if c0:
            #     context = c0
            #     self.logger.debug2(
            #         f'(d{settings.depth}) Iter {i}: changing context to {context}')

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
            contextPath = None

            # Trying move
            reachable, y0, a0 = model.reachable(attemptMove, context=context)
            # print('>', reachable, attemptMove, y0, a0)
            # model.npForward(a0, context, debug=True)
            # print(move)
            # print(nearestMove)
            # print(y0)
            # print(euclidean((nearestNode.pos + y0).plain(), goal.plain()))
            # print(reachable)

            newPos = nearestNode.pos + y0
            dist = euclidean(newPos.plain(), goal.plain())
            if not reachable or dist > mindist:
                # Attempt nearest move
                if dist > mindist:
                    reachable = False
                    self.logger.debug2(
                        f'(d{settings.depth}) Iter {i}: we are way too far from subgoal! Using {nearestMove} instead ({dist} > {mindist})')
                else:
                    self.logger.debug2(
                        f'(d{settings.depth}) Iter {i}: move {attemptMove} not reachable, our nearest move is {nearestMove} (after {j+1} attempt(s))')

                # print('invalid')
                if tryDirectMove:
                    reachable, y0, a0 = model.reachable(
                        nearestMove, context=context)
                    # print('REACHAAAAABLE?', reachable, y0, a0)
                    # print('>>', reachable, nearestMove, y0, a0)

            if not reachable:
                # Attempt 1/2 nearest move
                nearestMoveHalf = nearestMove * 0.5
                reachable, y0, a0 = model.reachable(
                    nearestMoveHalf, context=context)
                # print('REACHAAAAABLE?', reachable, y0, a0)
                # print('>>>', reachable, nearestMoveHalf, y0, a0)

            if not reachable and model.contextSpace:
                c0 = model.contextSpace.getPoint(nearestMoveId)[0]
                self.logger.debug2(
                    f'(d{settings.depth}) Iter {i}: move still not reachable, trying to reach a different context {c0}')

                reachable, y0, a0 = model.reachable(
                    nearestMoveHalf, context=c0)
                
                if reachable:
                    self.logger.debug2(
                        f'(d{settings.depth}) Iter {i}: move reachable with new context, is the context reachable?')
                    
                    try:
                        contextPath = self.plan(c0, nearestNode.state, settings=settings)
                    except ActionNotFound:
                        self.logger.debug(
                            f'(d{settings.depth}) Iter {i}: context {c0} is not reachable')
                        reachable = False

            if not reachable:
                directGoal = False
                self.logger.debug2(
                    f'(d{settings.depth}) Iter {i}: not reachable!!')
                attemptedUnreachable.append(newPos.plain())
                lastInvalids.append(newPos.plain())
                continue
            # print('valid')
            # print(a0)
            # print(y0)

            # # print('dist', dist)
            # if dist > mindist:
            #     # print('BOEZJPOIDVJPDVS *****')
            #     self.logger.debug2(
            #         f'(d{settings.depth}) Iter {i}: we are way too far from subgoal! {dist} > {mindist}')
            #     reachable, y0b, a0b = model.reachable(
            #         nearestMove, context=context)
            #     # print('>>', reachable, nearestMove, y0)
            #     if reachable:
            #         y0 = y0b
            #         a0b = a0b

            # Creating a new node
            newState = nearestNode.state.copy().apply(a0, self.dataset) if state else None
            if settings.dontMoveSpaces and dist > ignoreConstraintDistanceLimit:
                difference = newState.difference(nearestNode.state)
                dmChanged = False
                for dmSpace in settings.dontMoveSpaces:
                    self.logger.warning(f'{newState.context().projection(dmSpace)} vs {state.context().projection(dmSpace)} by doing {a0}')
                    dmDiff = difference.projection(dmSpace)
                    if dmDiff.norm() > self.MAX_MOVE_UNCHANGED_SPACES:
                        dmChanged = True
                        break
                if dmChanged:
                    directGoal = False
                    self.logger.debug2(
                        f'(d{settings.depth}) Iter {i}: move {attemptMove} is reachable but it affects {dmSpace}: {dmDiff} that shouldnt be changed')
                    attemptedBreakConstraint.append(newPos.plain())
                    lastInvalids.append(newPos.plain())
                    continue

            self.logger.debug2(
                f'(d{settings.depth}) Iter {i}: move {attemptMove} is usable')

            newNode = PathNode(pos=newPos, action=a0, goal=y0, model=model, parent=nearestNode,
                               state=newState)
            newNode.context = contextPath
            nodePositions[len(nodes)] = newPos.npPlain()
            nodes.append(newNode)
            self.logger.debug2(
                f'(d{settings.depth}) Iter {i}: node {newNode} attached to {nearestNode}')

            # Distance
            dist = euclidean(newPos.plain(), goal.plain())
            if dist > mindist:
                directGoal = False
            if dist < mindist:
                mindist = dist
                if dist < maxdistIncomplete:
                    minnode = newNode

            # print(dist)
            self.logger.debug2(
                f'(d{settings.depth}) Iter {i}: distance from goal {dist:.3f} (max {maxdist:.3f})')
            if dist < maxdist:
                self.logger.debug2(
                    f'(d{settings.depth}) Iter {i}: close enough to the goal {dist:.3f} (max {maxdist:.3f})')

                # print('FOUND')
                path = newNode.createPath(goal)
                break
        
            newPos = nearestNode.pos + y0
            if lastPos and np.sum(np.abs(newPos.npPlain() - lastPos)) < 1.:
                self.logger.debug2(
                    f'(d{settings.depth}) Iter {i}: deadlock, we are not moving enough! Stopping here')
                break
            lastPos = newPos.plain()

        if minnode and not path:
            path = minnode.createPath(goal)

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
        self._plotNodes(nodes, goal, attemptedUnreachable, attemptedBreakConstraint)
        # if not path:
        #     # print('Failed!')
        #     p1 = None
        #     if settings.allowContextPlanning:
        #         # print('Trying context planning')
        #         c0, p0, d0 = self._contextPlanning(
        #             model, goal, state=state, settings=settings)
        #         if p0:
        #             lastState = p0[0][-1].state
        #             # print('Successful context planning')
        #             settings.controlledSpaces = settings.controlledSpaces[:-1]

        #             newSettings = settings.clone()
        #             newSettings.allowContextPlanning = False
        #             p1, d1 = self.planDistance(
        #                 goal, state=lastState, settings=newSettings)
        #             # print('RESULT?')
        #             # print(p1)
        #             if p1:
        #                 mindist = d1
        #                 paths = p0.extends(p1)
        #                 path = paths[0].nodes()
        #                 # print(path)
        #                 # print('^^^^^^^^')
        #         # print(mindist)
        #         # print(model)
        #     if not p1:
        #         self.logger.warning(
        #             f"(d{settings.depth}) Planning failed for goal {goal}")
        #         raise ActionNotFound(f'Failed to create a path to reach {goal}', mindist if mindist < math.inf else None)
        # # else:
        # #     print('Yes!')

        self.logger.debug(
            f"(d{settings.depth}) Planning {'generated' if path else 'failed'} for goal {goal} in {i+1} step(s)")

        if hierarchical:
            self.logger.debug2(
                f"(d{settings.depth}) Planning sub level actions...")
            anyNonPrimitive = False
            for node in path:
                if not node.action.space.primitive():
                    anyNonPrimitive = True
                    # print('NODDE?')
                    node.execution = self.plan(
                        node.action, state=state)
            if anyNonPrimitive:
                self.logger.debug2(
                    f"(d{settings.depth}) Planning sub level actions... [Finished]")
            else:
                self.logger.debug2(
                    f"(d{settings.depth}) ...not needed, all actions are primitives")
        # print('Miaou==')

        # if settings.depth == 0:
        #     print('OVEERRRRR')
        #     print(path)

        print(path)

        return path, max(0, mindist)

    def _contextPlanning(self, model, goal, state=None, settings=PlanSettings()):
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
                c0, state=state, settings=settings)
        except Exception:
            return None, None, None
        print('--------')
        print(cpath[0][-1].state)
        print(c0)

        return c0, cpath, cdistance

        # model.inverse()
    
    def _plotNodes(self, nodes, goal, attemptedUnreachable, attemptedBreakConstraint):
        import matplotlib.pyplot as plt
        goalPlain = goal.plain()
        # plt.figure()
        if attemptedUnreachable:
            attemptedUnreachable = np.array(attemptedUnreachable)
            plt.scatter(attemptedUnreachable[:, 0], -attemptedUnreachable[:, 1], marker='.', color='black')
        if attemptedBreakConstraint:
            attemptedBreakConstraint = np.array(attemptedBreakConstraint)
            plt.scatter(attemptedBreakConstraint[:, 0], -attemptedBreakConstraint[:, 1], marker='o', color='purple')
        plt.scatter(goalPlain[0], -goalPlain[1], marker='x', color='orange')
        for node in nodes:
            if node.parent is not None:
                pos = node.pos.plain()
                parentPos = node.parent.pos.plain()
                plt.plot([parentPos[0], pos[0]], [-parentPos[1], -
                                                  pos[1]], f"{'b' if node.valid else 'r'},-")
        for node in nodes:
            if node.parent is None:
                pos = node.pos.plain()
                plt.scatter(pos[0], -pos[1], marker='x', color='blue')
        #plt.scatter(goalPlain[0], goalPlain[1], 'x', color='purple')
        #plt.draw()
        plt.show()
