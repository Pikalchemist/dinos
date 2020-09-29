import numpy as np
import random
import copy
import math

import time

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from ..agent import Agent

from exlab.utils.io import parameter
from exlab.modular.module import manage

from dino.utils.move import MoveConfig
# from ...utils.maths import uniformSampling, iterrange

# from ...data.data import InteractionEvent
# from ...data.dataset import Dataset
# from ...models.regression import RegressionModel
# from ...planners.planner import Planner

from dino.agents.tools.datasets.dataset import Dataset
from dino.agents.tools.strategies.strategy_set import StrategySet


class Learner(Agent):
    """Default Learner, learns by episode but without any choice of task and goal, choose strategy randomly."""

    DATASET_CLASS = Dataset

    def __init__(self, environment, dataset=None, performer=None, planner=None, options={}):
        """
        dataset Dataset: dataset of the agent
        strategies Strategy list: list of learning strategies available to the agent
        env Environment: environment of the experiment
        """
        dataset = parameter(dataset, self.DATASET_CLASS())
        super().__init__(environment, dataset=dataset, performer=performer,
                         planner=planner, options=options)
        manage(dataset).attach(self)

        # if self.dataset:
        #     self.addChildModule(self.dataset)

        self.trainStrategies = StrategySet(agent=self)
        # self.reachStrategies.append(reachStrategies if reachStrategies else None)#AutonomousExploration(self))

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
    
    def addEvent(self, event, config, cost=1.):
        if self.dataset:
            self.dataset.addEvent(event, cost=cost)

    def trainable(self):
        return True

    def train(self, iterations=None, untilIteration=None, episodes=None, untilEpisode=None):
        self.schedule(self._trainSchedule, iterations=iterations,
                      untilIteration=untilIteration, episodes=episodes, untilEpisode=untilEpisode)
    
    def _trainSchedule(self, iterations=None, untilIteration=None, episodes=None, untilEpisode=None):
        """Runs the learner until max number of iterations."""
        goalIteration = untilIteration if untilIteration else (
            self.iteration + iterations if iterations else None)
        goalEpisode = untilEpisode if untilEpisode else (
            self.episode + episodes if episodes else None)
        while ((goalIteration is None or self.iteration < goalIteration) and
               (goalEpisode is None or self.episode < goalEpisode)):
            self.syncCounter()
            self._train()
            self.syncCounter()

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

        self._postEpisode(memory, config)

    def _performEpisode(self, config):
        # Run an episode of the given strategy
        if config.strategy not in self.trainStrategies:
            raise Exception('{config.strategy} is not avaiable within {self}')
        return config.strategy.run(config)

    def _preEpisode(self):
        # Choose learning strategy randomly
        strategy = self.trainStrategies.sample()
        config = MoveConfig(strategy=strategy)

        self.logger.debug(f'Strategy used at iteration {self.iteration}: {config.strategy}', tag='strategy')
        return config

    def _postEpisode(self, memory, config):
        # self.logger.info('Adding episode of length {} to the dataset'
        #                  .format(len(memory)), 'DATA')

        if not config.evaluating:
            for event in memory:
                self.addEvent(event, config)
    
    def _postTest(self, memory, config):
        print('Done.')
        if not config.evaluating:
            for event in memory:
                self.addEvent(event, config)
    # Api
    # def apiget_time(self, range_=(-1, -1)):
    #     return {'data': iterrange(self.iterationTimes, range_)}
