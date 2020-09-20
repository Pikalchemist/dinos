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
from .planner import Planner

import numpy as np
from scipy.spatial.distance import euclidean

import itertools
import random
import math
import copy


class RRTPlanner(Planner):
    """
    Plans action sequences to reach given goals.
    This Planner can use both:
    - model hierarchy: to find primitive actions correponding to a goal
    - same model skill chaining: to reach out of range goals in a given model
    """

    def __init__(self, dataset, env=None, hierarchical=True, chaining=True):
        self.env = env
        self.dataset = dataset
        self.hierarchical = hierarchical
        self.chaining = chaining
        self.trees = {}
        self.semmap = None

    def partitions(self, goal, settings=PlanSettings()):
        settings = settings.clone()

        parts = goal.flat()
        space = goal.space
        # delegated = [[p, [p.space], 1] for p in parts if p.space.delegateto]
        for i in range(1):  # TODO  id:12
            # Find models related to our goal
            # models [(model, goal, related_parts, prob), ...]
            parts_left = list(parts)
            partition = []
            while parts_left:
                models = [[m, m.reachesSpaces([p.space for p in parts_left]), 0] for m in self.dataset.models
                          if settings.freeSpace(space)]
                models = [[m, list(p), prob]
                          for m, p, prob in models if len(p) > 0]
                # models += delegated
                for m in models:
                    m[2] = m[0].competence()  # float(len(m[1]))

                if not models:
                    raise ActionNotFoundException(f'No model found planning how to reach {goal}')
                models = np.array(models)[np.argsort(m[2])]
                model = models[0]
                # model = uniformRowSampling(models, [prob for _, _, prob in models])

                parts_left = [p for p in parts_left if p.space not in model[1]]
                # if isinstance(model[0], Model):  # Inverse model
                partition.append(
                    (model[0], Goal(*[p for p in parts if p.space in model[1]]), model[1]))
                # else:  # Delegated space
                #     partition += self.partitions(self.dataset.delegate_goal(model[0]))
        return partition


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
                f'Failed to create a path to reach {goal}. Space {goal.space} trying to be controlled twice!', None)
        settings.controlledSpaces += model.outcomeSpace
        print(settings.controlledSpaces)

        hierarchical = hierarchical if hierarchical is not None else self.hierarchical
        space = model.outcomeSpace
        if self.env:
            state = state if state else self.env.state(self.dataset)
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
                raise ActionNotFoundException(f'Failed to create a path to reach {goal}', mindist if mindist < math.inf else None)
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

    def _plotNodes(self, nodes, goal):
        import matplotlib.pyplot as plt
        goalPlain = goal.plain()
        # plt.figure()
        plt.scatter(goalPlain[0], -goalPlain[1], marker='x', color='orange')
        for node in nodes:
            if node.parent is not None:
                pos = node.pos.plain()
                parentPos = node.parent.pos.plain()
                plt.plot([parentPos[0], pos[0]], [-parentPos[1], -pos[1]], f"{'b' if node.valid else 'r'},-")
        for node in nodes:
            if node.parent is None:
                pos = node.pos.plain()
                plt.scatter(pos[0], -pos[1], marker='x', color='blue')
        #plt.scatter(goalPlain[0], goalPlain[1], 'x', color='purple')
        #plt.draw()
        plt.show()

    '''def __plan(self, model, goal, hierarchical=-1, length=0, compute_actions=True):
        hierarchical = hierarchical if hierarchical != -1 else self.hierarchical
        goal = goal.get_projection(model.outcomeSpace)

        ids, dists = model.outcomeSpace.nearestDistance(goal)
        node = model.outcomeSpace.getPoint(ids)[0]

        if self.env:
            y0, a0, c0 = model.inverse(node, context=model.currentContext(self.env))[0:3]
        else:
            y0, a0, c0 = model.inverse(node)[0:3]
        data = PathNode(action=a0, goal=y0, model=model)
        node = y0

        if hierarchical:
            if not a0.space.primitive():
                data.paths = self.plan(a0, hierarchical=hierarchical)

                # print("Need hierarchy {}".format(paths))

        path = Path([data])

        if dists[0] > 0.2 and length < 4 and self.chaining:  # TODO not fixed value id:5
            new_goal = goal - node

            g, dist = self.__plan(model, new_goal, hierarchical=hierarchical, length=length+1)
            if dist > dists[0]:
                return g, dists[0]
            return g + path, dist
        else:
            return path, dists[0]'''


def test_paths():
    a2a = Path([PathNode(SingleAction(0, "a2a1")),
                PathNode(SingleAction(0, "a2a2"))])
    a2b = Path([PathNode(SingleAction(0, "a2b"))])
    a2 = PathNode(SingleAction(0, "a2"))
    a2.paths = [a2a, a2b]
    a = Path([PathNode(SingleAction(0, "a1")), a2])
    b = Path([PathNode(SingleAction(0, "b1")), PathNode(
        SingleAction(0, "b2")), PathNode(SingleAction(0, "b3"))])
    paths = Paths([a, b])
    #print(paths)
    al = paths.getActionList()
    print(al)
    print("A")
    #print(a.getActionList())
    pass


def test():
    from data.dataset import Dataset
    from data.data import Goal, SingleData, InteractionEvent, ActionList, Action, SingleAction, Observation, SingleObservation
    from models.model import Model
    from data.space import SingleSpace
    import random
    print("PlannerTest")
    timethis(first=True)

    dataset = Dataset()

    a = SingleSpace({'min': np.array(
        [-100, -100]), 'max': np.array([100, 100])}, "A", effector=True)
    b = SingleSpace(
        {'min': np.array([-100, -100]), 'max': np.array([100, 100])}, "B")
    d = SingleSpace(
        {'min': np.array([-100, -100]), 'max': np.array([100, 100])}, "D")

    m = Model([a], [b])
    m2 = Model([b], [d])

    dataset.set([a, b, d], [m, m2])

    timethis("Model creation")

    events = []
    for i in range(5000):
        event = InteractionEvent()
        rx = random.uniform(0, 50)
        ry = random.uniform(0, 50)
        event.actions = ActionList(Action(SingleAction(a, [rx, ry])))
        event.outcomes = Observation(SingleObservation(
            b, [rx, ry]), SingleObservation(d, [rx, ry]))
        events.append(event)

    tests = []
    tests.append(Goal(SingleData(d, [190., 90.])))
    #N = 10
    #for i in xrange(N):
    #    tests.append(Goal(SingleData(d, [random.uniform(0, 90), random.uniform(0, 90)])))

    '''import utils
    events, tests = utils.load_raw("tempdata")

    timethis("Load")'''

    for event in events:
        dataset.addEvent(event)

    timethis("Point addition")

    print(a.number)
    print(d.number)

    #data = [events, tests]
    #utils.save_raw(data, "tempdata")

    #a = np.random.rand(5000, 2)
    #ids = np.random.randint(5000, size=5000)

    r = range(0, 5000, 2)

    timethis("Save")

    #print(len(d.lids))

    #print(d.number)
    #for i in xrange(100):
    #    d.getLid(r)

    #b = a[r]
    #np.argwhere(ids == r)

    '''for v in tests:
        y, a = m2.actionSpace.nearest(v)'''

    planner = Planner(dataset)

    for v in tests:
        print("Test hierarchical")
        #print(planner.planModel(m2, v).getActionList())
        paths = planner.plan(v, hierarchical=True)
        #print(paths)
        print(paths.getActionList())
        print("Competence: {}".format(dataset.get_competence_std_paths(paths)))
        print("Test non hierarchical")
        paths = planner.plan(v, hierarchical=False).getActionList()
        print(paths)
        #y, a = m2.bestLocality(v)[0:2]
    #print(y, a)

    timethis("Estimate goal ({})".format(len(tests)))

    #from evaluation.visualization import Visualizer
    #vis = Visualizer(m2.get_action_outcomes_visualizer())
    #vis.plot()

    #timethis("Plot")
    #return dataset


if __name__ == '__main__':
    test_paths()
    test()
