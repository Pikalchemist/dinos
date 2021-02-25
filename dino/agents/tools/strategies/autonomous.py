import random
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
import copy
import time

from dino.utils.move import MoveConfig

from dino.utils.maths import sigmoid, threshold, linearValue

from dino.data.data import Action, ActionList
# from dino.data.data import *
from dino.data.space import Space
from dino.data.path import ActionNotFound, Path

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
        self.randomThreshold = self.options.get(
            'randomThreshold', 0.1)  # 1 -> only random
        self.randomFirstPassStart = 1.
        self.randomFirstPassEnd = 0.4
        self.randomFirstPassNumber = 150
        self.randomFirstPassNumberMin = 20

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(self, []))
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

    def _runIteration(self, config):
        if config.goal:
            for _ in range(2):
                if self.reachGoalContext(config):
                    if self.exploreGoal(config):
                        return
                else:
                    break
        self.testRandomAction(config)

    def exploreGoal(self, config):
        # assert config.exploitation is False
        assert config.depth == 0

        # for _ in self.performer.iterative():
        # Choose between local and global exploration
        config.absoluteGoal = config.goal.absoluteData(self.agent.environment.state())
        probUseRandom, path = self.useGoalCriteria(config.goal, config)
        useGoal = random.uniform(0, 1) > probUseRandom
        config.result.randomProbability = probUseRandom

        self.logger.debug(f'goal exploration decision: criteriaRandom={probUseRandom}->useGoal={useGoal} exploration={config.exploitation} path={path}')

        if useGoal:  # We have chosen local optimizattion
            self.testPath(path, config)
            config.result.reachedGoal = self.agent.environment.state().context().projection(config.goal.space)
            config.result.reachedGoalDistance = (config.result.reachedGoal - config.absoluteGoal).norm()
            config.result.reachedGoalStatus = 'goal attempted'
        elif not config.exploitation:  # We have chosen random exploration
            if config.goalContext and not config.changeContext:
                config.changeContext = True
                return False
            else:
                self.testRandomAction(config)
                config.result.reachedGoalStatus = 'random chosen instead'
        else:
            self.testRandomAction(config, zero=True)
            config.result.reachedGoalStatus = 'zero chosen because exploitation'

        return True

    def useGoalCriteria(self, goal, config):
        """Criteria used to choose between local and global exploration (Straight from Matlab SGIM code)."""
        prob = 1.0

        # First pass: only random
        if goal.space.number < self.randomFirstPassNumberMin:
            return 30., Path()

        probFirstPass = linearValue(self.randomFirstPassStart, self.randomFirstPassEnd,
                                    (goal.space.number - self.randomFirstPassNumberMin) / self.randomFirstPassNumber)
        # print(probFirstPass)
        if not config.exploitation and random.uniform(0, 1) < probFirstPass:
            return 20. + probFirstPass, Path()  # Random action

        # Try planning to goal
        try:
            path, _, distance = self.planner.planDistance(goal, model=config.model, settings=config.plannerSettings)
        except ActionNotFound:
            path = Path()
        # print(path)
        if not path:
            config.result.reachedGoal = 'planning failed'
            self.logger.warning(
                f"Planning failed for goal {goal}, switching to random")
            return 10., Path()  # Random action

        # Compute criteria
        if config.exploitation:
            prob = -1.
        else:
            error = distance / (goal.space.maxDistance * 0.1)
            # (distance - space.options['err']) / space.options['range']
            # prob = 0.8*(sigmoid(error) - 0.5) + 0.5
            # prob = (prob - self.randomFirstPass) / (1. - self.randomFirstPass)
            prob = error  # threshold(prob, self.randomThreshold)

        '''space = goal.space
        if len(space.data) > self.options['min_points']:
            _, dist = space.nearest(goal, 5)
            if len(dist) < 5:
                return prob
            mean_dist = np.mean(dist)
            x = (mean_dist - space.options['err']) / space.options['range']
            prob = 0.8*(sigmoid(x) - 0.5) + 0.5'''
        # prob = 0.9*(sigmoid(x) - 0.5) + 0.5   # From matlab SGIM code
        return prob, path

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
