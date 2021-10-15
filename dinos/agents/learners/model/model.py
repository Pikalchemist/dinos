import numpy as np
import random
import copy
import math

import time

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from ..learner import Learner

from exlab.utils.io import parameter
from dinos.utils.move import MoveConfig
# from ....utils.maths import uniformSampling, iterrange

# from ....data.data import InteractionEvent
# from ....data.dataset import Dataset
# from ....models.regression import RegressionModel
# from ....planners.planner import Planner
from dinos.agents.tools.datasets.model_dataset import ModelDataset
from dinos.agents.tools.models.regression import RegressionModel


class ModelLearner(Learner):
    MODEL_CLASS = RegressionModel
    DATASET_CLASS = ModelDataset
    MULTI_EPISODE = 1

    def __init__(self, environment, dataset=None, performer=None, planner=None, options={}):
        dataset = parameter(dataset, self.DATASET_CLASS(modelClass=self.MODEL_CLASS))
        super().__init__(environment, dataset, performer, planner, options)

    # def _preEpisode(self):
    #     # Choose learning strategy randomly
    #     strategy = self.trainStrategies.sample()
    #     config = MoveConfig(strategy=strategy)

    #     self.logger.debug('Strategy used at iteration {}: {}'.format(
    #         self.iteration, config.strategy), 'STRAT')
    #     return config

    # def _postEpisode(self, memory):
    #     super()._postEpisode(memory)
