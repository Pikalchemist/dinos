import numpy as np
import random
import copy
import math

import time

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from ..agent import Agent

from exlab.utils.io import parameter
from exlab.interface.graph import Graph

from dino.utils.move import MoveConfig
# from ...utils.maths import uniformSampling, iterrange

# from ...data.data import InteractionEvent
# from ...data.dataset import Dataset
# from ...models.regression import RegressionModel
# from ...planners.planner import Planner

from dino.agents.tools.datasets.dataset import Dataset
from dino.agents.tools.strategies.strategy_set import StrategySet
from dino.agents.tools.policies.policy import LearningPolicy
from dino.environments.priors.maps.auto import Mapper


class Learner(Agent):
    """Default Learner, learns by episode but without any choice of task and goal, choose strategy randomly."""

    DATASET_CLASS = Dataset

    def __init__(self, host, dataset=None, performer=None, planner=None, options={}):
        """
        dataset Dataset: dataset of the agent
        strategies Strategy list: list of learning strategies available to the agent
        env Environment: host of the experiment
        """
        dataset = parameter(dataset, self.DATASET_CLASS())
        super().__init__(host, dataset=dataset, performer=performer,
                         planner=planner, options=options)
        self.dataset.attachLearner(self)

        self.learningPolicy = LearningPolicy.EACH_ITERATION
        self.processedEvents = []

        # if self.dataset:
        #     self.addChildModule(self.dataset)

        self.trainStrategies = StrategySet(agent=self)
        # self.reachStrategies.append(reachStrategies if reachStrategies else None)#AutonomousExploration(self))

        self.featureMap = Mapper.get(self.environment.__class__)

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(
            self, ['dataset', 'trainStrategies', 'processedEvents']))
        return dict_
    
    def findStrategy(self, name):
        return next((strategy for strategy in self.trainStrategies if strategy.name == name), None)

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
    
    def addEvent(self, event, config, cost=1., convertToDataset=True):
        if self.dataset and not config.evaluating and event.iteration not in self.processedEvents:
            if convertToDataset:
                event.convertTo(spaceManager=self.dataset, toData=True)
            self.processedEvents.append(event.iteration)
            self._addEvent(event, config, cost)
    
    def _addEvent(self, event, config, cost=1.):
        self.dataset.addEvent(event, cost=cost)
    
    def preProcessEvent(self, event, config, cost=1., convertToDataset=False):
        if self.dataset and event.iteration not in self.processedEvents:
            if convertToDataset:
                event.convertTo(spaceManager=self.dataset, toData=True)
            self._preProcessEvent(event, config, cost)
    
    def _preProcessEvent(self, event, config, cost=1.):
        pass
    
    def addMemory(self, memory, config):
        memory = self._convertMemory(memory)

        for event in memory:
            self.preProcessEvent(event, config)

        if not config.evaluating:
            for event in memory:
                self.addEvent(event, config, convertToDataset=False)
            self.dataset.updated()
    
    def _convertMemory(self, memory):
        memory = list(memory)
        if self.dataset:
            for event in memory:
                event.convertTo(spaceManager=self.dataset, toData=True)
        return memory

    def trainable(self):
        return True

    def train(self, iterations=None, untilIteration=None, episodes=None, untilEpisode=None):
        self.syncCounter()
        self.iterationType[self.iteration] = 'train'
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
        self.iterationType[self.iteration] = 'end'

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
        self.configs[self.iteration] = config

        # Performs the episode
        memory = self._performEpisode(config)

        self._postEpisode(memory, config)

    def _performEpisode(self, config):
        # Run an episode of the given strategy
        if config.strategy not in self.trainStrategies:
            raise Exception(f'{config.strategy} is not avaiable within {self}')
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
        self.addMemory(memory, config)
    
    def _postTest(self, memory, config):
        self.addMemory(memory, config)

    class ResultAnalyser(Agent.ResultAnalyser):
        @property
        def trainingConfigs(self):
            return {i: c for i, c in self.agent.configs.items() if c.training}
        
        def reachedSpaces(self, data, key=None):
            if key is None:
                key = lambda item: item.space
            if isinstance(data, dict):
                data = data.values()
            return list(set([key(item) for item in data]))
        
        def groupBySpace(self, data, key=None):
            if key is None:
                key = lambda item: item.space

            spaces = self.reachedSpaces(data, key=key)
            if isinstance(data, dict):
                return {space: {i: item for i, item in data.items() if key(item) == space} for space in spaces}
            else:
                return {space: [item for item in data if key(item) == space] for space in spaces}

        def _filterNoDict(self, data, iteration=True):
            if not iteration:
                return list(data.values())
            return data
        
        def _filterString(self, data):
            return data and not isinstance(data, str)

        # Analyse
        def randomActions(self, range_=None, iteration=True):
            points = {i: config.result.action for i, config in self.trainingConfigs.items() if config.result.action}
            return self._filterNoDict(points, iteration)

        def goals(self, range_=None, iteration=True):
            points = {i: config.goal for i, config in self.trainingConfigs.items() if config.goal}
            return self._filterNoDict(points, iteration)

        def contextGoals(self, range_=None, iteration=True):
            points = {i: config.goalContext for i, config in self.trainingConfigs.items() if config.goalContext}
            return self._filterNoDict(points, iteration)
        
        def reachedGoals(self, range_=None, iteration=True):
            points = {i: config.result.reachedGoal for i, config in self.trainingConfigs.items() if self._filterString(config.result.reachedGoal)}
            return self._filterNoDict(points, iteration)
        
        def reachedContextGoals(self, range_=None, iteration=True):
            points = {i: config.result.reachedContext for i, config in self.trainingConfigs.items() if self._filterString(config.result.reachedContext)}
            return self._filterNoDict(points, iteration)
        
        def goalErrors(self, range_=None, iteration=True):
            points = {i: (config.goal, config.result.reachedGoal - config.absoluteGoal)
                      for i, config in self.trainingConfigs.items() if config.absoluteGoal and self._filterString(config.result.reachedGoal)}
            return self._filterNoDict(points, iteration)

        # Visual
        def visualizeGeneralData(self, data, title='', color=False, options={}):
            gs = []
            grouped = self.groupBySpace(data)
            for space, data in grouped.items():
                g = Graph(title=f'{title} from {space}', options=options)
                g.scatter(list(data.values()), color=list(data.keys()) if color else None)
                gs.append(g)
            return gs

        def visualizeRandomActions(self, range_=None, color=False, options={}):
            return self.visualizeGeneralData(self.randomActions(range_=range_), title='Random actions', color=color, options=options)

        def visualizeGoals(self, range_=None, color=False, options={}):
            return self.visualizeGeneralData(self.goals(range_=range_), title='Goals', color=color, options=options)
        
        def visualizeContextGoals(self, range_=None, color=False, options={}):
            return self.visualizeGeneralData(self.contextGoals(range_=range_), title='ContextGoals', color=color, options=options)
        
        def visualizeReachedGoals(self, range_=None, color=False, options={}):
            return self.visualizeGeneralData(self.reachedGoals(range_=range_), title='Reached Goals', color=color, options=options)

        def visualizeReachedContextGoals(self, range_=None, color=False, options={}):
            return self.visualizeGeneralData(self.reachedContextGoals(range_=range_), title='Reached ContextGoals', color=color, options=options)

        def visualizeGoalErrors(self, range_=None, color=False, options={}):
            title = 'Goal Errors'
            data = self.goalErrors(range_=range_)

            gs = []
            grouped = self.groupBySpace(data, key=lambda item: item[0].space)
            for space, data in grouped.items():
                values = [(d[0].plain(), d[1].plain()) for d in data.values()]
                g = Graph(title=f'{title} from {space}', options=options)
                g.arrow(values, color=list(data.keys()) if color else None)
                gs.append(g)
            return gs

    # Api
    # def apiget_time(self, =(-1, -1)):
    #     return {'data': iterrange(self.iterationTimes, range_)}
