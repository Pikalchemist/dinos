import numpy as np
import random
import copy
import math

import time

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from ..learner import Learner

from dino.utils.move import MoveConfig
# from ....utils.maths import uniformSampling, iterrange

# from ....data.data import InteractionEvent
# from ....data.dataset import Dataset
# from ....models.regression import RegressionModel
# from ....planners.planner import Planner


class ModelLearner(Learner):
    MODEL_CLASS = RegressionModel
    MULTI_EPISODE = 1

    def __init__(self, environment, dataset=None, performer=None, planner=None, options={}):
        dataset = dataset if dataset else Dataset.deserialize(options.get('dataset', {}),
                                                              modelClass=self.MODEL_CLASS)
        super().__init__(environment, dataset, performer, planner, options)

    def addEventToDataset(self, event, config):
        self.dataset.addEvent(event)

    def _trainEpisode(self):
        # memory = []
        # for i in range(self.MULTI_EPISODE):
        #     config = self._preEpisode()
        #
        #     # Performs the episode
        #     memoryStrategy = self._performEpisode(config)
        #     InteractionEvent.incrementList(memoryStrategy, len(memory))
        #     memory += memoryStrategy

        config = self._preEpisode()

        # Performs the episode
        memory = self._performEpisode(config)

        # Add memory to dataset
        self.iteration += len(memory)
        self.set_iteration(self.iteration)
        for event in memory:
            event.convertTo(spaceManager=self.dataset, toData=True)

        self.logger.info('Adding episode of length {} to the dataset'
                         .format(len(memory)), 'DATA')

        for event in memory:
            self.addEventToDataset(event, config)

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
