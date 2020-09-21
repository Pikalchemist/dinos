import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
import copy
import time

from dino.utils.move import MoveConfig

from dino.utils.maths import sigmoid, threshold

from dino.data.data import Action, ActionList
# from dino.data.data import *
from dino.data.space import Space
from dino.data.path import ActionNotFound, Paths

from .random import RandomStrategy


class AutonomousStrategy(RandomStrategy):
    """AutonomousStrategy exploration strategy. Used in SAGG-RIAC algorithm."""

    def __init__(self, agent, name=None, performer=None, planner=None, options={}):
        """
        environment EnvironmentV4: environment of the learning agent
        dataset Dataset: episodic memory of the agent using the strategy (needed for local exploration)
        name string: name of the given strategy
        options dict: options of the strategy
        verbose boolean: indicates the level of debug info
        """
        super().__init__(agent, name=name, planner=planner, performer=performer,
                         options=options)
        # Contains the following keys:
        #   'min_points': min number of iterations before enabling the use of local exploration
        #   'nb_iteration': number of iterations per learning episode
        # Store method chosen for every iteration with the strategy, stored by learning episode
        self.methods = []
        # Store local global criteria for every iteration with the strategy, stored by learning episode
        self.criteria = []
        self.randomThreshold = self.options.get(
            'randomThreshold', 0.1)  # 1 -> only random
        self.randomFirstPass = self.options.get(
            'randomFirstPass', 0.1)  # 1 -> only random

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(
            self, [], exportPathType=True))
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, agent, options=None, obj=None):
    #     obj = obj if obj else cls(agent, dict_.get(
    #         'name'), options=dict_.get('options', {}))
    #     obj = RandomStrategy._deserialize(
    #         dict_, agent, options=options, obj=obj)
    #     return obj

    def _preRun(self, config):
        super()._preRun(config)

        # Add a list with the choices of methods for the episode
        self.methods.append([])
        self.criteria.append([])

    def _runIteration(self, config):
        # timethissub(3)
        self.reachGoalContext(config)
        if config.goal:
            self.exploreGoal(config)
        else:
            self.testRandomAction(config)
        # timethissub(3, "All exploreGoal")

    def exploreGoal(self, config):
        # assert config.exploitation is False
        assert config.depth == 0
        goal = config.goal
        paths = None

        for _ in self.performer.iterative():
            # Choose between local and global exploration
            # timethissub(1)
            probUseGoal, paths = self.useGoalCriteria(goal, config)
            useGoal = random.uniform(0, 1) > probUseGoal
            self.logger.debug(
                f'goal exploration decision: criteria={probUseGoal}->{useGoal} exploration={config.exploitation} paths={paths}')
            # print(paths)
            # timethissub(1, "Init localGlobalCriteria")
            # if not config.exploitation:
            #     self.criteria[-1].append(probUseGoal)
            #     self.methods[-1].append(1 if useGoal else 0)

            # print('Goal: {}'.format(goal))
            if useGoal:  # We have chosen local optimizattion
                # print("Using Local optimization after random for n=" + str(self.n) + ", ~probUseGoal: " + str(probUseGoal))
                # print('----Paths----')
                # print(goal)
                # print('---')
                # print(paths)
                self.testGoal(goal, paths, config)

            else:  # We have chosen random exploration
                # if random.uniform(0, 1) <= 0.3 or not config.model:
                #     actionSpaces = self.agent.dataset.controllableSpaces()
                # else:
                #     actionSpaces = self.agent.dataset.controllableSpaces(config.model.actionSpace.iterate())
                # , actionSpaces=actionSpaces
                self.testRandomAction(config)

        return paths

    def useGoalCriteria(self, goal, config):
        """Criteria used to choose between local and global exploration (Straight from Matlab SGIM code)."""
        prob = 1.0

        # First pass: only random
        probFirstPass = 0. # TODO threshold(self.randomFirstPass, self.randomThreshold)
        print(probFirstPass)
        if not config.exploitation and random.uniform(0, 1) < probFirstPass:
            return 1., Paths()  # Random action

        # Try planning to goal
        try:
            paths, distance = self.planner.planDistance(goal, model=config.model)
        except ActionNotFound:
            self.logger.warning(f"Planning failed for goal {goal}, switching to random")
            return 1., Paths()  # Random action

        # Compute criteria
        if config.exploitation:
            prob = -1.
        else:
            space = goal.space
            # (distance - space.options['err']) / space.options['range']
            x = distance / space.maxDistance
            prob = 0.8*(sigmoid(x) - 0.5) + 0.5
            prob = (prob - self.randomFirstPass) / (1. - self.randomFirstPass)
            prob = threshold(prob, self.randomThreshold)

        '''space = goal.space
        if len(space.data) > self.options['min_points']:
            _, dist = space.nearest(goal, 5)
            if len(dist) < 5:
                return prob
            mean_dist = np.mean(dist)
            x = (mean_dist - space.options['err']) / space.options['range']
            prob = 0.8*(sigmoid(x) - 0.5) + 0.5'''
        # prob = 0.9*(sigmoid(x) - 0.5) + 0.5   # From matlab SGIM code
        return prob, paths

    # def runLocalOptimization(self, a, goal, initial_simplex=None, config=None):
    #     """Perform local exploration using Nelder-Mead optimization method."""

    #     # Function to minimize for Nelder-Mead
    #     # if config.exploitation else (self.options['nb_iteration'] - self.n) # TODO  id:10
    #     maxiter = 1
    #     aspace = a.get_space()

    #     def function(x):
    #         return self.testNelderMead(x, aspace, goal, config)

    #     minimize(function, a.flatten(), method=custom_nelder_mead, options={'xtol': 1e-3, 'ftol': 1e-6,
    #                                                                         'maxiter': maxiter, 'maxfev': maxiter,
    #                                                                         'initial_simplex': initial_simplex})

    # def testNelderMead(self, a_data, aspace, g, config):
    #     """Method called when testing an action with Nelder-Mead algorithm."""

    #     a = ActionList(aspace.asTemplate(a_data.tolist()))

    #     self.test_action(a, config)
    #     space = g.get_space()
    #     dist = space.options['out_dist']/space.options['max_dist']
    #     '''if not config.exploitation:
    #         for i in range(len(self.memory[-1][-1][3])):
    #             if self.memory[-1][-1][3][i] == space.id:
    #                 dist = min(euclidean(g, self.memory[-1][-1][2][i]), space.options['out_dist']) / space.options['max_dist']'''
    #     return dist
