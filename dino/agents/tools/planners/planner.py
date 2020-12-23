'''
    File name: planner.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

from exlab.modular.module import Module
from exlab.utils.io import parameter

# from ..utils.debug import timethis
from dino.utils.maths import linearValue
from dino.data.data import Data, Goal, SingleAction, Action, ActionList
from dino.data.path import ActionNotFound, Path, PathNode
from dino.data.space import SpaceKind
# from dino.models.model import Model
from dino.utils.result import Result

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
    def __init__(self, computeActions=True, performing=False):
        self.computeActions = computeActions
        self.length = 0
        self.depth = -1
        self.context = False

        self.controlledSpaces = []
        self.dontMoveSpaces = []

        self.allowContextPlanning = True
        self.forceContextPlanning = False
        self.tryContextPlanning = True
        self.mayUseContext = False
        self.hierarchical = None
        self.maxIterations = None

        self.performing = performing

        self.result = Result(None)

    def freeSpace(self, space):
        for s in self.controlledSpaces:
            if s.intersects(space):
                return False
        return True

    def clone(self, **kwargs):
        obj = copy.copy(self)
        obj.controlledSpaces = list(self.controlledSpaces)
        obj.dontMoveSpaces = list(self.dontMoveSpaces)
        for key, value in kwargs.items():
            setattr(obj, key, value)
        return obj


Part = namedtuple('Part', ['model', 'goal'])


class Planner(Module):
    """
    Plans action sequences to reach given goals.
    This Planner can use both:
    - model hierarchy: to find primitive actions correponding to a goal
    - same model skill chaining: to reach out of range goals in a given model
    """

    MAX_DISTANCE = 0.01
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

    def __plan(self, model, goal, state=None, settings=PlanSettings()):
        planning = Planning(self, model, goal, state, settings)
        return planning.execute()

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


class Planning(object):
    def __init__(self, planner, model, goal, state, settings):
        self.planner = planner
        self.settings = settings

        self.logger.debug(
            f'=== New planning {self.logTag()} === -> {goal} using {model}')# with context {state}

        self.model = model
        self.space = model.outcomeSpace
        self.space._validate()

        self.startState = parameter(state, self.planner.environment.state(self.planner.dataset))
        self.startPosition = self.startState.context().projection(self.space)
        self.goal = goal.projection(self.space).relativeData(self.startState)
        self.checkSpaceIsControllable(self.goal, self.space)

        self.logger.debug(
            f'== Relative goal is {self.goal} (starting pos {self.startPosition})')

        self.initParameters()
        self.initVariables()

    def logTag(self):
        return f'(d{self.settings.depth})'

    @property
    def logger(self):
        return self.planner.logger

    def initParameters(self):
        self.maxIter = parameter(self.settings.maxIterations, 20)
        self.maxIterNoNodes = 5
        if self.settings.dontMoveSpaces:
            self.maxIter *= 10
            self.maxIterNoNodes *= 4

        self.maxDistanceBest = self.model.getPrecision(Planner.MAX_DISTANCE, 0.5)
        self.maxDistance = self.maxDistanceBest * 3 + 0.02 * self.goal.norm()
        self.maxDistanceIncomplete = self.maxDistanceBest * 8 + 0.1 * self.goal.norm()

        self.contextPlanningPossible = True
        if self.model.contextSpace and not self.settings.freeSpace(self.model.contextSpace.convertTo(kind=SpaceKind.BASIC)):
            self.contextPlanningPossible = False
            self.logger.debug(
                f'{self.logTag()} Context planning wont be available as {self.model.contextSpace} is already controlled')
    
    def initVariables(self):
        self.minDistanceReached = self.goal.norm()
        self.closestNodeToGoal = None

        self.nodes = [PathNode(pos=self.space.zero(), absPos=self.startPosition, state=self.startState)]
        # nodePositions = np.zeros((maxIter + 1, space.dim))

        self.path = Path()
        # self.dist = 0
        self.directGoal = True
        self.currentContextFailures = 0
        # previousMove = None
        # print('GOAL: {}'.format(goal))
        # print('================')
        self.lastPosition = None

        # 
        self.invalidPointRatioLimit = 0.05
        self.ignoreConstraintDistanceLimit = 0.05 * self.space.maxDistance

        self.attemptedUnreachable = []
        self.attemptedBreakConstraint = []
        self.lastInvalids = []

        # Attempt results
        self.a0, self.y0, self.c0, self.distance0 = [None, None, None, None]
    
    def execute(self):
        if self.space._number < 100 or self.minDistanceReached < self.maxDistanceBest:
            return Path(), None, 0

        # Main loop
        self.i = -1
        self.iterationOffset = 0
        while self.i < self.maxIter + self.iterationOffset:
            self.i += 1

            if len(self.nodes) <= 1 and self.i > self.maxIterNoNodes or len(self.nodes) <= 2 and self.i > self.maxIterNoNodes * 2:
                self.logger.warning(f'Not enough nodes, aborting...')
                break
    
            # Generating distant subgoals
            self.subGoal = self.generateSubGoal()
        
            nearestNode = self.nearestNode(self.subGoal)
            self.state = nearestNode.state
            context = self.state.context()
            if self.planner.semmap:
                context = self.planner.semmap.context(context)
            
            self.logger.debug2(
                f'{self.logTag()} Iter {self.i}: chosen subGoal is {self.subGoal} (direct={self.directGoal}, final goal {self.goal})\n    Nearest node is: {nearestNode.pos}')#\n    With context {self.state}

            baseMove = self.subGoal - nearestNode.pos

            reachable, move = self.searchWithCurrentContext(baseMove, context)
            if reachable is None:
                self.logger.debug2(
                    f'{self.logTag()} Iter {self.i}: invalid move {self.y0}, skipping iteration')
                self.directGoal = False
                self.iterationOffset += 1
                if move is True:
                    self.currentContextFailures += 1
                continue
            reachable, contextPath = self.searchWithDifferentContext(baseMove, reachable)

            # self.logger.info(f'{self.logTag()} Iter {self.i}: end {reachable}')
            if not reachable:
                self.directGoal = False
                self.logger.debug2(
                    f'{self.logTag()} Iter {self.i}: not reachable!!')
                nearestNode.failures += 1
                if move:
                    self.attemptedUnreachable.append((nearestNode.pos + move).plain() + nearestNode.pos.plain())
                    self.lastInvalids.append((nearestNode.pos + move).plain())
                continue
        
            newNode = self.addNode(nearestNode, contextPath)
            if not newNode:
                continue

            if self.checkDistance(newNode):
                break
        
        if self.closestNodeToGoal and not self.path:
            self.path = self.closestNodeToGoal.createPath(self.goal, self.pathSettings())

        # self._plotNodes(self.nodes, self.goal, self.space, self.attemptedUnreachable, self.attemptedBreakConstraint)

        self.logger.debug(
            f"{self.logTag()} Planning {'generated' if self.path else 'failed'} for goal {self.goal} in {self.i+1} step(s)")
        
        # if self.path:
            # self.path = self.simplifyPath(self.path)
            # self._plotNodes(self.path.nodes, self.goal, self.space, self.attemptedUnreachable, self.attemptedBreakConstraint)
        
        if not self.settings.performing and not self.settings.context and self.settings.depth == 0:
            self.settings.result.planningSuccess = self.path is not None
            self.settings.result.planningSteps = self.i

        minDistance = max(0, self.minDistanceReached)

        # if not self.path and minDistance > 0:
        #     self.model.updatePrecision(False, min(minDistance, self.space.maxDistance * 0.2))

        finalState = self.path[-1].state if self.path else None
        return self.path, finalState, minDistance

    def searchWithCurrentContext(self, baseMove, context):
        riskyMove, move, safeMove, nearestMove, _ = self.findBestMove(baseMove, context)

        maxIncorrectMoveDistance = 0.01 * self.space.maxDistance * self.model.limitMoves
        maxNoMoveDistance = 0.15 * self.space.maxDistance * self.model.limitMoves
        if self.minDistanceReached < self.space.maxDistance:
            maxNoMoveDistance *= self.minDistanceReached / self.space.maxDistance
        maxNoMoveDistanceMin = maxNoMoveDistance * 2

        # self.logger.error(
        #     f'{self.logTag()} Iter {self.i}: {maxNoMoveDistance} {maxNoMoveDistanceMin} {self.minDistanceReached}')

        if move is None:
            self.logger.error(
                f'{self.logTag()} Iter {self.i}: failed to find a move!\n{self.model} {self.goal} {context}')
            return None, None

        if move.norm() < self.space.maxDistance * 0.001:
            return False, None

        self.logger.debug2(
            f'{self.logTag()} Iter {self.i}: chosen move is {move} (risky move is {riskyMove}) {maxNoMoveDistance} {maxNoMoveDistanceMin}')

        reachable = False
        if not self.settings.forceContextPlanning:
            if riskyMove:
                if riskyMove.norm() < maxNoMoveDistance and self.minDistanceReached > maxNoMoveDistanceMin:
                    return None, True
                reachable = self.attempt(riskyMove, context) if riskyMove else False

            if not reachable:
                if move.norm() < maxNoMoveDistance and self.minDistanceReached > maxNoMoveDistanceMin:
                    return None, True
                reachable = self.attempt(move, context)
            # if reachable:
            #     newPosition = nearestNode.pos + y0
            #     absPos = startPos + newPosition
            #     dist = newPosition.distanceTo(attemptMove)

            if not reachable:
                self.currentContextFailures += 1

            if not reachable and safeMove:
                reachable = self.attempt(safeMove, context)
            
            for multiplier in (1., 0.5):
                if not reachable:
                    reachable = self.attempt(move * multiplier, context)

            if reachable and self.y0.norm() < maxIncorrectMoveDistance and self.minDistanceReached > maxIncorrectMoveDistance * 2:
                return False, None
            if reachable and self.y0.norm() < maxNoMoveDistance and self.minDistanceReached > maxNoMoveDistanceMin:
                return None, None
            
            # if reachable and baseMove.norm() > 0.01:
            #     if self.y0.norm() < 0.01:
            #         reachable = False
            #     else:
            #         # diff = (subGoal - y0).norm() / subGoal.norm()
            #         orientation = 1. - baseMove.npPlain().dot(self.y0.npPlain()) / (baseMove.norm() * self.y0.norm())
            #         if orientation > 0.4: # or y0.norm() < 0.2 * goal.norm() or y0.norm() > 5.0 * goal.norm():
            #             reachable = False
            #             self.logger.info(
            #                 f'{self.logTag()} Iter {self.i}: Move orientation is off compared to baseMove!\nOrt: {orientation}, y0: {self.y0}, baseMove: {baseMove}, move: {move}')
        # self.logger.info(f'{self.logTag()} Iter {self.i}: {reachable}')
        return reachable, move
    
    def searchWithDifferentContext(self, baseMove, reachableWithContext):
        reachable = reachableWithContext
        contextPath = None
        settingsTestContext = self.settings.forceContextPlanning or self.settings.tryContextPlanning
        if (settingsTestContext or not reachableWithContext) and self.model.contextSpace and self.contextPlanningPossible:
            # Save with context results
            a0wc, y0wc, d0wc = [self.a0, self.y0, self.distance0]

            riskyMove, move, safeMove, nearestMove, c0 = self.findBestMove(baseMove)
            c0 = self.findBestContext(move)
            riskyMove, move, safeMove, nearestMove, _ = self.findBestMove(baseMove, c0)

            # c0 = self.model.contextSpace.getPoint(nearestMoveId)[0]
            if settingsTestContext:
                self.logger.debug2(f'{self.logTag()} Iter {self.i}: testing with context {c0}')
            else:
                self.logger.debug2(f'{self.logTag()} Iter {self.i}: move still not reachable, trying to use a different context {c0}')

            reachable = self.attempt(riskyMove, c0, adaptContext=True) if riskyMove else False

            if not reachable:
                reachable = self.attempt(move, c0, adaptContext=True)
            # if reachable:
            #     newPosition = nearestNode.pos + y0
            #     absPos = startPos + newPosition
            #     dist = newPosition.distanceTo(attemptMove)

            if not reachableWithContext or self.currentContextFailures >= self.maxIter * 0.1:
                if not reachable and safeMove:
                    reachable = self.attempt(safeMove, c0, adaptContext=True)
                
                for multiplier in (1., 0.5, 0.4, 0.25, 0.15):
                    if not reachable:
                        reachable = self.attempt(move * multiplier, c0, adaptContext=True)

            # print('===')
            # print(reachable, y0, a0)

            switchBackWithContext = False
            if reachableWithContext and (not reachable or d0wc <= self.distance0 + self.maxDistanceBest):
                switchBackWithContext = True
            elif reachable:
                self.logger.debug(
                    f'{self.logTag()} Iter {self.i}: choose to {"change" if reachableWithContext else "use"} context ({d0wc} > {self.distance0 + self.maxDistanceBest}), is the context reachable?')

                newSettings = self.settings.clone(context=True)
                # if self.i == 0:
                newSettings.dontMoveSpaces.append(self.space)
                # First context planning
                # if i == 0:
                #     newSettings.maxIterations = 100
                contextPath, contextState = self.attemptPlanningContext(c0, self.state, newSettings)

                # Try without dontMoveSpaces constraint
                # if not contextPath:
                #     newSettings.dontMoveSpaces = []
                #     contextPath, contextState = self.attemptPlanningContext(c0, self.state, newSettings)

                if contextPath:
                    c0 = contextState.context()
                    reachable = self.attempt(nearestMove, c0)

                    if not reachable:
                        reachable = self.attempt(nearestMove * 0.5, c0)
                else:
                    reachable = False

                if not reachable:
                    switchBackWithContext = True
                    self.logger.debug(
                        f'{self.logTag()} Iter {self.i}: context {c0} is not reachable')
                elif reachableWithContext and d0wc <= self.distance0 + self.maxDistanceBest * 0.5 * len(contextPath):
                    switchBackWithContext = True
            
            if switchBackWithContext and reachableWithContext:
                self.logger.debug2(
                    f'{self.logTag()} Iter {self.i}: switching back to not modifying the context')
                reachable, self.a0, self.y0, self.distance0 = [True, a0wc, y0wc, d0wc]
                contextPath = None

            if contextPath:
                self.state = contextState
                self.logger.debug(
                    f'{self.logTag()} Iter {self.i}: changing context to reach c0={c0} with a {len(contextPath)} step sub path')

        return reachable, contextPath
    
    def addNode(self, nearestNode, contextPath):
        newPosition = nearestNode.pos + self.y0
        absPosition = self.startPosition + newPosition
        goalDistance = euclidean(newPosition.plain(), self.goal.plain())

        # Creating a new node
        newState = self.nextState(nearestNode, goalDistance, self.state)
        if newState is None:
            return None

        self.logger.debug2(
            f'{self.logTag()} Iter {self.i}: move is usable ({self.a0}->{self.y0}), future state: {newState} ({absPosition})')
        
        self.lastInvalids = []
        # if len(lastInvalids) > 5:
        #     lastInvalids = lastInvalids[-5:]
        # if lastInvalids:
        #     lastInvalids = lastInvalids[1:]

        newNode = PathNode(pos=newPosition, absPos=absPosition, action=self.a0, goal=self.y0, model=self.model, parent=nearestNode,
                            state=newState)
        newNode.context = contextPath
        # nodePositions[len(nodes)] = newPosition.npPlain()
        self.nodes.append(newNode)
        self.logger.debug2(
            f'{self.logTag()} Iter {self.i}: node {newNode} attached to {nearestNode}')
        return newNode
    
    def nextState(self, nearestNode, goalDistance, state):
        newState = state.copy().apply(self.a0, self.planner.dataset)
        if self.settings.dontMoveSpaces and goalDistance > self.ignoreConstraintDistanceLimit:
            if self.hasSpacesChanged(nearestNode, newState):
                self.directGoal = False
                nearestNode.failures += 1
                self.attemptedBreakConstraint.append((nearestNode.pos + self.y0).plain() + nearestNode.pos.plain())
                self.lastInvalids.append((nearestNode.pos + self.y0).plain())
                return None
        return newState

    def checkDistance(self, newNode):
        # Distance
        distance = euclidean(newNode.pos.plain(), self.goal.plain())
        self.logger.debug2(
            f'{self.logTag()} Iter {self.i}: distance from goal {distance:.3f} (max {self.maxDistance:.3f})')

        if distance > self.minDistanceReached:
            self.directGoal = False
            if self.minDistanceReached < self.maxDistance and self.closestNodeToGoal:
                self.logger.debug(f'{self.logTag()} Iter {self.i}: close enough to the goal {self.minDistanceReached:.3f} (max {self.maxDistance:.3f})')
                self.path = self.closestNodeToGoal.createPath(self.goal, self.pathSettings())
                if not self.settings.performing and not self.settings.context and self.settings.depth == 0:
                    self.settings.result.planningDistance = (self.minDistanceReached, self.maxDistance)
                return True
        else:
            self.minDistanceReached = distance

        # print(distance)
        if distance < self.maxDistanceBest:
            self.logger.debug(f'{self.logTag()} Iter {self.i}: close enough to the goal {distance:.3f} (max best {self.maxDistanceBest:.3f})')
            self.path = newNode.createPath(self.goal, self.pathSettings())
            if not self.settings.performing and not self.settings.context and self.settings.depth == 0:
                self.settings.result.planningDistance = (distance, self.maxDistanceBest)
            return True
        elif distance < self.maxDistanceIncomplete:
            self.closestNodeToGoal = newNode
    
        # newPosition = nearestNode.pos + y0
        # if lastPos and np.sum(np.abs(newPosition.npPlain() - lastPos)) < 1.:
        #     self.logger.debug2(
        #         f'{self.logTag()} Iter {self.i}: deadlock, we are not moving enough! Stopping here')
        #     break
        self.lastPosition = newNode.pos.plain()
        return False
    
    def simplifyPath(self, path):
        self.i = 0
        nodes = path.nodes
        index = 0
        maxJump = 4
        while index < len(nodes) - 2:
            node = path[index]
            state = node.parent.state
            self.logger.debug2(f'Simplify: trying a jump from {index}/{len(path)}')
            for jump in range(maxJump, 0, -1):
                if index + jump >= len(path) - 1:
                    continue
                # check if there is a context change
                if any(path[index + i].context for i in range(jump + 2)):
                    self.logger.debug2(f'Simplify: context!')
                    continue

                self.logger.debug2(f'Simplify: trying a jump +{jump} from {index}/{len(path)}')

                # Pre move
                preMove = node.goal
                for i in range(1, jump + 1):
                    preMove += path[index + i].goal
                if not self.attempt(preMove, state.context()):
                    continue
            
                newPosition = node.parent.pos + self.y0
                absPosition = self.startPosition + newPosition
                goalDistance = euclidean(newPosition.plain(), self.goal.plain())
                newState = self.nextState(node.parent, goalDistance, state)
                if newState is None:
                    continue
                newPreNode = PathNode(pos=newPosition, absPos=absPosition, action=self.a0, goal=self.y0, model=self.model, parent=node.parent,
                                      state=newState)
                
                self.logger.debug2(f'Simplify: pre jump possible, reaching {newPosition} instead of {path[index + jump].pos} by doing {self.a0}=>{self.y0}')

                # Post Move
                # preA0 = self.a0
                # preY0 = self.y0
                postMove = path[index + jump + 1].goal + preMove - self.y0
                if not self.attempt(postMove, state.context()):
                    continue

                newPosition = newPreNode.pos + self.y0
                absPosition = self.startPosition + newPosition
                goalDistance = euclidean(newPosition.plain(), self.goal.plain())
                newState = self.nextState(newPreNode, goalDistance, newPreNode.state)
                if newState is None:
                    continue
                newPostNode = PathNode(pos=newPosition, absPos=absPosition, action=self.a0, goal=self.y0, model=self.model, parent=newPreNode,
                                       state=newState)
                
                self.logger.debug2(f'Simplify: post jump possible, reaching {newPosition} instead of {path[index + jump + 1].pos} by doing {self.a0}=>{self.y0} ({path[index + jump + 1].goal})')
                self.logger.debug2(f'Simplify: compacting {jump+1}->2')

                nodes = nodes[0:index] + [newPreNode, newPostNode] + nodes[index + jump + 1:]
                break
            index += 1

        return path

    def attempt(self, move, context, adaptContext=False):
        precision = 0.45 + (self.i / self.maxIter) * 0.05
        precisionOrientation = 0.05 + (self.i / self.maxIter) * 0.05
        if adaptContext:
            context = self.findBestContext(move)
        reachable, self.a0, self.y0, self.distance0 = self.model.reachable(
            move, context=context, precision=precision, precisionOrientation=precisionOrientation, dontMove=self.settings.dontMoveSpaces)
        contextLog = f' with context {context}' if adaptContext else ''
        self.logger.debug2(f'{self.logTag()} Iter {self.i}: testing {move}{contextLog}: {"reachable" if reachable else "not reachable"}')
        return reachable
    
    def attemptPlanningContext(self, context, state, settings):
        try:
            return self.planner.plan(context.setRelative(False), state, settings=settings)
        except ActionNotFound:
            return None, None
    
    def checkSpaceIsControllable(self, goal, space):
        if not self.settings.freeSpace(space):
            error = f'Failed to create a path to reach {goal}. Space {space} trying to be controlled twice! (controlled spaces: {self.settings.controlledSpaces})'
            self.logger.debug(error)
            raise ActionNotFound(error, None)
        self.settings.controlledSpaces += space
    
    def pathSettings(self):
        settings = self.settings.clone()
        settings.controlledSpaces.remove(self.space)
        return settings
    
    def hasSpacesChanged(self, nearestNode, newState):
        difference = newState.difference(nearestNode.state)
        for space in self.settings.dontMoveSpaces:
            spaceDifference = difference.projection(space)
            if spaceDifference.norm() > self.planner.MAX_MOVE_UNCHANGED_SPACES:
                # self.logger.warning(f'{newState.context().projection(space)} vs {state.context().projection(dmSpace)} by doing {a0}')
                self.logger.debug2(
                    f'{self.logTag()} Iter {self.i}: move is reachable but it affects {space}: {spaceDifference} that shouldnt be changed')
                return True
        return False
    
    def generateSubGoal(self):
        # Finding a subgoal
        if random.uniform(0, 1) <= self.planner.GOAL_SAMPLE_RATE or self.directGoal:
            subGoal = self.goal
        else:
            subGoal = self.generateRandomPoint()
            # space.goal([random.uniform(minrand, maxrand) for x in range(space.dim)])

            if self.lastInvalids:
                invalids = np.array(self.lastInvalids)
                for _ in range(100):
                    distanceToNearestNode = (self.nearestNode(self.subGoal).pos - subGoal).norm()
                    dists = np.sum((invalids - subGoal.npPlain()) ** 2, axis=1) ** 0.5
                    if not np.any(dists < self.invalidPointRatioLimit * self.space.maxDistance) and distanceToNearestNode > 0.2 * self.space.maxDistance:
                        break
                    subGoal = self.generateRandomPoint()
                    # self.logger.debug2(f'(d{settings.depth}) Iter {i}: Too close to invalid point, retrying...')
        return subGoal
    
    def generateRandomPoint(self):
        origin = random.uniform(0., 1.) * self.goal.npPlain()
        # point = [random.uniform(0., 1.) for _ in range(space.dim)]
        # point = point * self.goal
        width = 1.5 * min(0.05 * self.space.maxDistance + self.goal.norm(), self.space.maxDistance)
        point = origin + width * np.array([random.uniform(-1., 1.)
                                   for _ in range(self.space.dim)])
        return self.space.goal(point)
    
    def nearestNode(self, point):
        pointPlain = point.plain()
        dlist = np.sum(
            (np.array([node.pos.plain() for node in self.nodes]) - pointPlain) ** 2, axis=1) ** 0.5 * np.array([node.penalty() for node in self.nodes])
        return self.nodes[np.argmin(dlist)]
    
    def findBestMove(self, baseMove, context=None):
        # lastVariance = -1.
        # We are looking for a valid point in the outcome space
        # self.logger.debug2(f'=====================')

        ratio = (self.space._number - 100) / 300
        number = int(linearValue(1, 2, ratio))
        close = linearValue(1., 0.05, ratio)
        riskyMove, _, _ = self._searchBestMove(baseMove, number, close, context)
        if riskyMove is None:
            return None, None, None, None, None
        
        number = int(linearValue(1, 6, ratio))
        close = linearValue(0.2, 0.05, ratio)
        move, nearestMove, ids = self._searchBestMove(baseMove, number, close, context)
        if move is None:
            return None, None, None, None, None
        
        if riskyMove.norm() < move.norm():
            riskyMove = None

        c0 = None
        if context is None:
            cs = self.model.contextSpace.getNpPlainPoint(ids)
            c0 = self.model.contextSpace.asTemplate(np.mean(cs, axis=0))

        safeMove = move
        if (move - baseMove).norm() < move.norm() * 0.1:
            move = baseMove
        else:
            move *= 0.9
        # self.logger.debug2(f'---------------------')
        return riskyMove, move, safeMove, nearestMove, c0  # , nearestMoveId
    
    def findBestContext(self, move):
        # ratio = (self.space._number - 100) / 300
        number = 1 #int(linearValue(1, 6, ratio))

        ids, _, _ = self.model.nearestOutcome(move, n=number, nearestUseContext=False)
        cs = self.model.contextSpace.getNpPlainPoint(ids)
        c0 = self.model.contextSpace.asTemplate(np.mean(cs, axis=0))
        # self.logger.debug2(
        #     f'HEY {ids} {self.model.outcomeSpace.getNpPlainPoint(ids)}')
        return c0
    
    def _searchBestMove(self, baseMove, number, close, context):
        move = baseMove.clone()
        for j in range(10):
            if j > 0:
                move *= 1. * distanceToCenter / move.norm()
            if j == 1:
                move *= self.model.limitMoves
            # self.logger.debug2(
            #     f'(d{settings.depth}) Iter {i} {j}: {move}')

            ids, _, _ = self.model.nearestOutcome(
                move, context=context, n=number, nearestUseContext=False)
            # ids, dists = self.model.outcomeContextSpace.nearestDistance(moveContext, n=5, restrictionIds=ids)
            if len(ids) == 0:
                return None
            nearestMoveId = ids[0]
            nearestMove = self.space.getPoint(nearestMoveId)[0]

            points = self.space.getNpPlainPoint(ids)
            center = np.mean(points, axis=0)
            distanceToCenter = np.sum(center ** 2) ** .5
            distanceMoveToCenter = np.mean(
                np.sum((center - move.npPlain()) ** 2) ** .5)
            # self.logger.debug2(f'(d{self.settings.depth}) Iter {i} {j}: {self.space.getPlainPoint(ids)}')
            # self.logger.debug2(f'(d{self.settings.depth}) Iter {i} {j}: {nearestMove} {distanceToCenter} {distanceToNode}')

            # if context is None:
            #     cs = self.model.contextSpace.getNpPlainPoint(ids)
            #     c0 = np.mean(cs, axis=0)
            # self.logger.info(f'{j}: {move} {distanceToCenter} {distanceMoveToCenter} {self.space.maxDistance * close}\n    {points}')

            if (j > 0 and distanceMoveToCenter <= self.space.maxDistance * close) or distanceToCenter > move.norm():
                break
        
        return move, nearestMove, ids

    def _plotNodes(self, nodes, goal, space, attemptedUnreachable, attemptedBreakConstraint):
        import matplotlib.pyplot as plt
        goalPlain = goal.plain()
        # plt.figure()
        points = np.array([self.generateRandomPoint().plain() for _ in range(1000)])
        plt.scatter(points[:, 0], -points[:, 1], marker='.', color='gray')
        if attemptedUnreachable:
            # attemptedUnreachable = np.array(attemptedUnreachable)
            # plt.scatter(attemptedUnreachable[:, 0], -attemptedUnreachable[:, 1], marker='.', color='black')
            # for x, y, fx, fy in attemptedUnreachable:
            for x, y, fx, fy in attemptedUnreachable:
                plt.plot([fx, x], [-fy, -y], 'm,--')
        if attemptedBreakConstraint:
            # attemptedBreakConstraint = np.array(attemptedBreakConstraint)
            # plt.scatter(attemptedBreakConstraint[:, 0], -attemptedBreakConstraint[:, 1], marker='o', color='purple')
            for x, y, fx, fy in attemptedBreakConstraint:
                plt.plot([fx, x], [-fy, -y], 'r,--')
        plt.scatter(goalPlain[0], -goalPlain[1], marker='X', color='orange')
        for node in nodes:
            if node.parent is not None:
                pos = node.pos.plain()
                parentPos = node.parent.pos.plain()
                plt.scatter(pos[0], -pos[1], marker='+', color='green' if node.valid else 'blue')
                plt.plot([parentPos[0], pos[0]], [-parentPos[1], -pos[1]], f"{'g' if node.valid else 'b'},-")
        for node in nodes:
            if node.parent is None:
                pos = node.pos.plain()
                plt.scatter(pos[0], -pos[1], marker='x', color='blue')
        #plt.scatter(goalPlain[0], goalPlain[1], 'x', color='purple')
        #plt.draw()
        plt.show()

