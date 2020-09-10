import numpy as np
import random
import copy
import math

import time

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from ..agent import Agent

from dino.utils.move import MoveConfig
# from ...utils.maths import uniformSampling, iterrange

# from ...data.data import InteractionEvent
# from ...data.dataset import Dataset
# from ...models.regression import RegressionModel
# from ...planners.planner import Planner

from dino.agents.tools.strategies.strategy_set import StrategySet


class Learner(Agent):
    """Default Learner, learns by episode but without any choice of task and goal, choose strategy randomly."""

    def __init__(self, environment, dataset=None, performer=None, planner=None, options={}):
        """
        dataset Dataset: dataset of the agent
        strategies Strategy list: list of learning strategies available to the agent
        env Environment: environment of the experiment
        """
        super().__init__(environment, performer=performer, options=options)
        self.dataset = dataset
        # if self.dataset:
        #     self.addChildModule(self.dataset)

        self.trainStrategies = StrategySet(agent=self)
        # self.reachStrategies.append(reachStrategies if reachStrategies else None)#AutonomousExploration(self))

        self.planner = planner if planner else Planner(self.dataset,
                                                       env=environment,
                                                       chaining=options.get(
                                                           "chaining", False),
                                                       hierarchical=options.get("hierarchical", True))

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(
            self, ['dataset', 'trainStrategies'], exportPathType=True))
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, environment, dataset=None, options={}, obj=None):
    #     from ...utils.loaders import DataManager

    #     obj = obj if obj else cls(
    #         environment, dataset, options=dict_.get('options', {}))
    #     obj = Agent._deserialize(dict_, environment, options=options, obj=obj)

    #     for strategy in dict_.get('trainStrategies', []):
    #         obj.trainStrategies.add(DataManager.loadType(strategy['path'], strategy['type'])
    #                                 .deserialize(strategy, obj, options=options))
    #     return obj

    def trainable(self):
        return True

    def train(self, iterations=None, untilIteration=None, episodes=None, untilEpisode=None):
        """Runs the learner until max number of iterations."""
        goalIteration = untilIteration if untilIteration else (
            self.iteration + iterations if iterations else None)
        goalEpisode = untilEpisode if untilEpisode else (
            self.episode + episodes if episodes else None)
        while ((goalIteration is None or self.iteration < goalIteration) and
               (goalEpisode is None or self.episode < goalEpisode)):
            self._train()

    def _train(self):
        self.trainEpisode()

    def trainEpisode(self):
        """Run one learning episode."""
        self.environment.setupEpisode()

        self._trainEpisode()

        self.iterationByEpisode.append(
            self.iteration - self.lastIterationEpisode)
        self.lastIterationEpisode = self.iteration
        self.episode += 1
        #self.measureIterationTime()

    def _trainEpisode(self):
        config = self._preEpisode()

        # Performs the episode
        memory = self._performEpisode(config)

        self._postEpisode(memory)

    def _performEpisode(self, config):
        # Run an episode of the given strategy
        if config.strategy not in self.trainStrategies:
            raise Exception("{} is not avaiable within {}".format(
                config.strategy, self))
        return config.strategy.run(config)

    def _preEpisode(self):
        # Choose learning strategy randomly
        strategy = self.trainStrategies.sample()
        config = MoveConfig(strategy=strategy)

        self.logger.debug('Strategy used at iteration {}: {}'.format(
            self.iteration, config.strategy), 'STRAT')
        return config

    def _postEpisode(self, memory):
        pass

    # Api
    # def apiget_time(self, range_=(-1, -1)):
    #     return {'data': iterrange(self.iterationTimes, range_)}