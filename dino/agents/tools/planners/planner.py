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
        self.forceContextPlanning = False
        self.mayUseContext = False
        self.hierarchical = None
        self.maxIterations = None

    def freeSpace(self, space):
        for s in self.controlledSpaces:
            if s.intersects(space):
                return False
        return True

    def clone(self):
        obj = copy.copy(self)
        obj.controlledSpaces = list(self.controlledSpaces)
        obj.dontMoveSpaces = list(self.dontMoveSpaces)
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
    GOAL_SAMPLE_RATE = 0.2

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
                for model in self.dataset.enabledModels():
                    spaces = list(model.reachesSpaces([p.space for p in partsLeft]))
                    if len(spaces) > 0 and settings.freeSpace(space):
                        models.append([model, spaces])

                if not models:
                    raise ActionNotFound(f'No model found planning how to reach {goal}')
                if len(models) == 1:
                    bestModel = models[0]
                else:
                    models = np.array(models)
                    models = models[np.argsort([-model[0].competence() for model in models])]
                    bestModel = models[0]
                    # model = uniformRowSampling(models, [prob for _, _, prob in models])

                partsLeft = [p for p in partsLeft if p.space not in bestModel[1]]
                partition.append(
                    Part(bestModel[0], Goal(*[p for p in parts if p.space in bestModel[1]])))
        return partition

    def plan(self, goal, state=None, model=None, settings=PlanSettings()):
        return self.planDistance(goal, state=state, model=model, settings=settings)[:2]

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
            path, finalState, dist = self.__plan(
                model, subgoal, state=state, settings=settings)
            totalDistance += dist
            paths.append(path)

        return paths[0], finalState, totalDistance

    def planActions(self, actionList, state=None, model=None, settings=PlanSettings()):
        settings = settings.clone()
        settings.depth += 1

        # print('---- planActions')
        actionList = actionList.convertTo(self.dataset)
        nodes = []
        for action in actionList:
            node = PathNode(action=action)
            # print(action)
            # print(action.space.primitive())
            if not action.space.primitive():
                node.execution, _ = self.plan(
                    action, state=state, model=model, settings=settings)
            # print(node.paths)
            nodes.append(node)
        # print('----')
        return Path(nodes)
    
    def generateRandomPoint(self, goal, space):
        origin = random.uniform(0., 1.) * goal.npPlain()
        # point = [random.uniform(0., 1.) for _ in range(space.dim)]
        # point = point * goal
        width = 0.7 * min(0.05 * space.maxDistance + goal.norm(), space.maxDistance)
        point = origin + width * np.array([random.uniform(-1., 1.)
                                   for _ in range(space.dim)])
        return space.goal(point)
    
    def findClosestMove(self, subgoal, nearestNode, model, context):
        lastVariance = -1.
        # We are looking for a valid point in the outcome space
        for j in range(10):
            if j == 0:
                move = subgoal - nearestNode.pos
            else:
                # subgoal = subgoaldistant
                move *= distancesToNode / move.norm()
            # self.logger.debug2(
            #     f'(d{settings.depth}) Iter {i} {j}: {move}')

            # moveContext = move.extends(context)
            ids, _, _ = model.nearestOutcome(
                move, context=context, n=5, nearestUseContext=False)
            # ids, dists = model.outcomeContextSpace.nearestDistance(moveContext, n=5, restrictionIds=ids)
            if len(ids) == 0:
                return None, None, 0
            nearestMove = model.outcomeSpace.getPoint(ids)[0]

            # print('----', j)
            # print(move)
            # print(nearestMove)
            # print(model.outcomeSpace.variance(ids))
            # print(model.outcomeSpace.denseEnough(ids))
            # print(model.outcomeSpace.maxDistance * 0.1)

            distancesToNode = 0.9 * np.mean(np.sum(np.array(
                model.outcomeSpace.getPlainPoint(ids)) ** 2, axis=1) ** .5)
            distancesToMove = np.mean(np.sum(np.array(
                model.outcomeSpace.getPlainPoint(ids) - move.npPlain()) ** 2, axis=1) ** .5)
            # self.logger.debug2(
            #     f'(d{settings.depth}) Iter {i} {j}: {model.outcomeSpace.getPlainPoint(ids)}')

            # self.logger.debug2(
            #     f'(d{settings.depth}) Iter {i} {j}: {nearestMove} {distancesToMove} {distancesToNode}')
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
        return move, nearestMove
    
    def findClosestMoveWithoutContext(self, subgoal, nearestNode, model):
        move = subgoal - nearestNode.pos
        ids, _, _ = model.nearestOutcome(move, n=1)
        return move, model.outcomeSpace.getPoint(ids)[0], ids[0]

    def __plan(self, model, goal, state=None, settings=PlanSettings()):
        self.logger.debug(f'=== New planning (d{settings.depth}) === -> {goal} using {model} with context {state}')

        # Checking if space already planned
        if not settings.freeSpace(goal.space):
            error = f'Failed to create a path to reach {goal}. Space {goal.space} trying to be controlled twice! (controlled spaces: {settings.controlledSpaces})'
            self.logger.debug(error)
            raise ActionNotFound(error, None)
        settings.controlledSpaces += model.outcomeSpace
        # print(goal.space)
        # print(settings.controlledSpaces)

        # Settings
        # hierarchical = parameter(settings.hierarchical, self.hierarchical)
        space = model.outcomeSpace
        # if self.environment:

        state = state if state else self.environment.state(self.dataset)
        goal = goal.projection(space)
        goal = goal.relativeData(state)
        startPos = state.context().projection(space)

        self.logger.debug(f'== Relative goal is {goal} (starting pos {startPos})')

        attemptedUnreachable = []
        attemptedBreakConstraint = []
        lastInvalids = []

        # parameters
        maxIter = parameter(settings.maxIterations, 20)
        if settings.dontMoveSpaces:
            maxIter *= 10
        maxIterNoNodes = 20

        # init variables
        maxdist = 2 + 0.02 * goal.norm()
        maxdistIncomplete = 5 + 0.1 * goal.norm()

        mindist = math.inf
        minnode = None

        zero = space.zero()
        nodes = [PathNode(pos=zero, absPos=startPos, state=state)]
        # nodePositions = np.zeros((maxIter + 1, space.dim))

        path = Path()
        dist = 0
        directGoal = True
        # previousMove = None
        # print('GOAL: {}'.format(goal))
        # print('================')
        lastPos = None

        invalidPointRatioLimit = 0.05
        ignoreConstraintDistanceLimit = 0.05 * space.maxDistance

        # Main loop
        for i in range(maxIter):
            # Finding a subgoal
            if random.uniform(0, 1) <= self.GOAL_SAMPLE_RATE or directGoal:
                subgoaldistant = goal
            else:
                subgoaldistant = self.generateRandomPoint(goal, space)
                # space.goal([random.uniform(minrand, maxrand) for x in range(space.dim)])

                if lastInvalids:
                    while True:
                        invalids = np.array(lastInvalids)
                        dists = np.sum((invalids - subgoaldistant.npPlain()) ** 2, axis=1) ** 0.5
                        if not np.any(dists < invalidPointRatioLimit * space.maxDistance):
                            break
                        subgoaldistant = self.generateRandomPoint(goal, space)
                        # self.logger.debug2(f'(d{settings.depth}) Iter {i}: Too close to invalid point, retrying...')
            
            if len(nodes) <= 3 and i > maxIterNoNodes:
                self.logger.warning(f'Not enough nodes, aborting...')
                break

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

            subgoal = subgoaldistant

            # Nearest Node
            subgoalPlain = subgoal.plain()
            dlist = np.sum(
                (np.array([node.pos.plain() for node in nodes]) - subgoalPlain) ** 2, axis=1) ** 0.5 * np.array([node.penalty() for node in nodes])
            nearestNode = nodes[np.argmin(dlist)]

            contextState = None
            state = nearestNode.state
            self.logger.debug2(
                f'(d{settings.depth}) Iter {i}: current state is {state}')
            context = state.context()
            if self.semmap:
                context = self.semmap.context(context)
            
            self.logger.debug2(
                f'(d{settings.depth}) Iter {i}: chosen subgoal is {subgoal}(->{subgoaldistant}) (direct={directGoal}) (final goal {goal}) with context {state}')

            move, nearestMove = self.findClosestMove(
                subgoal, nearestNode, model, context)

            if move is None:
                print(f'!!!!!!!! {model} {goal} {context}')
                continue

            self.logger.debug2(
                f'(d{settings.depth}) Iter {i}: corresponding chosen move is {move}')

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

            contextPath = None
            if not settings.forceContextPlanning:
                attemptMove = move
                tryDirectMove = True

                # Trying move
                reachable, y0, a0 = model.reachable(attemptMove, context=context)
                # print('>', reachable, attemptMove, y0, a0)
                # model.npForward(a0, context, debug=True)
                # print(move)
                # print(nearestMove)
                # print(y0)
                # print(euclidean((nearestNode.pos + y0).plain(), goal.plain()))
                # print(reachable)

                if reachable:
                    newPos = nearestNode.pos + y0
                    absPos = startPos + newPos
                    dist = euclidean(newPos.plain(), subgoal.plain())
                if not reachable or dist > mindist:
                    # Attempt nearest move
                    if dist > mindist:
                        reachable = False
                        self.logger.debug2(
                            f'(d{settings.depth}) Iter {i}: we are way too far from subgoal! Using {nearestMove} instead ({dist} > {mindist})')
                    else:
                        self.logger.debug2(
                            f'(d{settings.depth}) Iter {i}: move {attemptMove} not reachable, our nearest move is {nearestMove}')

                    # print('invalid')
                    if tryDirectMove:
                        attemptMove = nearestMove
                        reachable, y0, a0 = model.reachable(
                            attemptMove, context=context)
                        # print('REACHAAAAABLE?', reachable, y0, a0)
                        # print('>>', reachable, nearestMove, y0, a0)

                if not reachable:
                    # Attempt 1/2 nearest move
                    attemptMove = nearestMove * 0.5
                    reachable, y0, a0 = model.reachable(
                        attemptMove, context=context)
                    # print('REACHAAAAABLE?', reachable, y0, a0)
                    # print('>>>', reachable, nearestMoveHalf, y0, a0)
                
                if reachable and subgoal.norm() > 0.01:
                    if y0.norm() < 0.02:
                        reachable = False
                    else:
                        # diff = (subgoal - y0).norm() / subgoal.norm()
                        orientation = 1. - subgoal.npPlain().dot(y0.npPlain()) / (subgoal.norm() * y0.norm())
                        if orientation > 0.4 or y0.norm() < 0.2 * goal.norm() or y0.norm() > 5.0 * goal.norm():
                            reachable = False
                            self.logger.debug2(
                                f'(d{settings.depth}) Iter {i}: we\'re way too far from goal!')

            if (settings.forceContextPlanning or not reachable) and model.contextSpace:
                attemptMove, nearestMove, nearestMoveId = self.findClosestMoveWithoutContext(
                    subgoal, nearestNode, model)

                c0 = model.contextSpace.getPoint(nearestMoveId)[0]
                if settings.forceContextPlanning:
                    self.logger.debug2(
                        f'(d{settings.depth}) Iter {i}: trying to reach context {c0} first')
                else:
                    self.logger.debug2(
                        f'(d{settings.depth}) Iter {i}: move still not reachable, trying to use a different context {c0}')

                reachable, y0, a0 = model.reachable(
                    attemptMove, context=c0)

                if not reachable:
                    attemptMove = nearestMove
                    reachable, y0, a0 = model.reachable(
                        attemptMove, context=c0)

                if not reachable:
                    attemptMove = nearestMove * 0.5
                    reachable, y0, a0 = model.reachable(
                        attemptMove, context=c0)

                # print('===')
                # print(reachable, y0, a0)
                
                if reachable:
                    self.logger.debug2(
                        f'(d{settings.depth}) Iter {i}: move reachable with new context, is the context reachable?')

                    
                    newSettings = settings.clone()
                    if i == 0:
                        newSettings.dontMoveSpaces.append(goal.space)
                    # First context planning
                    # if i == 0:
                    #     newSettings.maxIterations = 100
                    try:
                        contextPath, contextState = self.plan(c0.setRelative(False), nearestNode.state, settings=newSettings)
                    except ActionNotFound:
                        contextPath = None
                    
                    # Try without dontMoveSpaces constraint
                    if not contextPath:
                        newSettings.dontMoveSpaces = []
                        try:
                            contextPath, contextState = self.plan(c0.setRelative(False), nearestNode.state, settings=newSettings)
                        except ActionNotFound:
                            contextPath = None

                    if contextPath:
                        c0 = contextState.context()
                        attemptMove = nearestMove
                        reachable, y0, a0 = model.reachable(
                            attemptMove, context=c0)

                        if not reachable:
                            attemptMove = nearestMove * 0.5
                            reachable, y0, a0 = model.reachable(
                                attemptMove, context=c0)

                    if not reachable:
                        self.logger.debug(
                            f'(d{settings.depth}) Iter {i}: context {c0} is not reachable')

            if not reachable:
                directGoal = False
                self.logger.debug2(
                    f'(d{settings.depth}) Iter {i}: not reachable!!')
                nearestNode.failures += 1
                attemptedUnreachable.append(subgoalPlain + nearestNode.pos.plain())
                lastInvalids.append(subgoalPlain)
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

            newPos = nearestNode.pos + y0
            absPos = startPos + newPos
            dist = euclidean(newPos.plain(), goal.plain())

            # Creating a new node
            if contextState:
                state = contextState.copy()
            else:
                state = state.copy()
            newState = state.apply(a0, self.dataset)
            if settings.dontMoveSpaces and dist > ignoreConstraintDistanceLimit:
                difference = newState.difference(nearestNode.state)
                dmChanged = False
                for dmSpace in settings.dontMoveSpaces:
                    dmDiff = difference.projection(dmSpace)
                    if dmDiff.norm() > self.MAX_MOVE_UNCHANGED_SPACES:
                        # self.logger.warning(f'{newState.context().projection(dmSpace)} vs {state.context().projection(dmSpace)} by doing {a0}')
                        dmChanged = True
                        break
                if dmChanged:
                    directGoal = False
                    self.logger.debug2(
                        f'(d{settings.depth}) Iter {i}: move {attemptMove} is reachable but it affects {dmSpace}: {dmDiff} that shouldnt be changed')
                    nearestNode.failures += 1
                    attemptedBreakConstraint.append(subgoalPlain + nearestNode.pos.plain())
                    lastInvalids.append(subgoalPlain)
                    continue

            self.logger.debug2(
                f'(d{settings.depth}) Iter {i}: move {attemptMove} is usable ({a0}->{y0}), future state: {newState} ({absPos})')
            
            lastInvalids = []
            # if len(lastInvalids) > 5:
            #     lastInvalids = lastInvalids[-5:]
            # if lastInvalids:
            #     lastInvalids = lastInvalids[1:]

            newNode = PathNode(pos=newPos, absPos=absPos, action=a0, goal=y0, model=model, parent=nearestNode,
                               state=newState)
            newNode.context = contextPath
            # nodePositions[len(nodes)] = newPos.npPlain()
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
            # if lastPos and np.sum(np.abs(newPos.npPlain() - lastPos)) < 1.:
            #     self.logger.debug2(
            #         f'(d{settings.depth}) Iter {i}: deadlock, we are not moving enough! Stopping here')
            #     break
            lastPos = newPos.plain()

        if minnode and not path:
            path = minnode.createPath(goal)

        # self._plotNodes(nodes, goal, space, attemptedUnreachable, attemptedBreakConstraint)

        self.logger.debug(
            f"(d{settings.depth}) Planning {'generated' if path else 'failed'} for goal {goal} in {i+1} step(s)")

        # if hierarchical:
        #     self.logger.debug2(
        #         f"(d{settings.depth}) Planning sub level actions...")
        #     anyNonPrimitive = False
        #     for node in path:
        #         if not node.action.space.primitive():
        #             anyNonPrimitive = True
        #             # print('NODDE?')
        #             node.execution = self.plan(
        #                 node.action, state=state)
        #     if anyNonPrimitive:
        #         self.logger.debug2(
        #             f"(d{settings.depth}) Planning sub level actions... [Finished]")
        #     else:
        #         self.logger.debug2(
        #             f"(d{settings.depth}) ...not needed, all actions are primitives")
        # print('Miaou==')

        # if settings.depth == 0:
        #     print('OVEERRRRR')
        #     print(path)

        # print(path)

        finalState = path[-1].state if path else None

        return path, finalState, max(0, mindist)

    # def _contextPlanning(self, model, goal, state=None, settings=PlanSettings()):
    #     settings = settings.clone()
    #     path = []
    #     print(model.controllableContext())
    #     if not model.controllableContext():
    #         return None, None, None

    #     c0 = model.adaptContext(goal, state.context())
    #     if not c0:
    #         return None, None, None

    #     try:
    #         cpath, _, cdistance = self.planDistance(
    #             c0, state=state, settings=settings)
    #     except Exception:
    #         return None, None, None
    #     print('--------')
    #     print(cpath[0][-1].state)
    #     print(c0)

    #     return c0, cpath, cdistance

        # model.inverse()
    
    def _plotNodes(self, nodes, goal, space, attemptedUnreachable, attemptedBreakConstraint):
        import matplotlib.pyplot as plt
        goalPlain = goal.plain()
        # plt.figure()
        points = np.array([self.generateRandomPoint(goal, space).plain() for _ in range(1000)])
        plt.scatter(points[:, 0], -points[:, 1], marker='.', color='gray')
        if attemptedUnreachable:
            # attemptedUnreachable = np.array(attemptedUnreachable)
            # plt.scatter(attemptedUnreachable[:, 0], -attemptedUnreachable[:, 1], marker='.', color='black')
            # for x, y, fx, fy in attemptedUnreachable:
            for x, y, fx, fy in attemptedUnreachable:
                plt.plot([fx, x], [-fy, -y], 'g,--')
        if attemptedBreakConstraint:
            # attemptedBreakConstraint = np.array(attemptedBreakConstraint)
            # plt.scatter(attemptedBreakConstraint[:, 0], -attemptedBreakConstraint[:, 1], marker='o', color='purple')
            for x, y, fx, fy in attemptedBreakConstraint:
                plt.plot([fx, x], [-fy, -y], 'r,--')
        plt.scatter(goalPlain[0], -goalPlain[1], marker='x', color='orange')
        for node in nodes:
            if node.parent is not None:
                pos = node.pos.plain()
                parentPos = node.parent.pos.plain()
                plt.plot([parentPos[0], pos[0]], [-parentPos[1], -
                                                  pos[1]], f"{'g' if node.valid else 'b'},-")
        for node in nodes:
            if node.parent is None:
                pos = node.pos.plain()
                plt.scatter(pos[0], -pos[1], marker='x', color='blue')
        #plt.scatter(goalPlain[0], goalPlain[1], 'x', color='purple')
        #plt.draw()
        plt.show()
