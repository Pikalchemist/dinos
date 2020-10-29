import numpy as np
import random
import copy
import math
import time
import threading

from exlab.modular.module import Module, manage
from exlab.lab.counter import AsyncCounter

from exlab.utils.io import parameter
from dino.utils.move import MoveConfig

from dino.agents.tools.planners.planner import Planner
from dino.agents.tools.performers.performer import Performer
from dino.agents.tools.strategies.strategy_set import StrategySet


# def make(id_, environment):
#     from ..utils.loaders import DataManager
#     return DataManager.makeAgent(id_, environment)


class Agent(Module):
    """An Agent performing actions in an environment.
    Can be subclassed to implement an algorithm designed to perform one or multiple tasks.

    Args:
        environment (Environment): The environment within which the agent will live and operate
        performer (Performer):
        options (dict): A dictionary of parameters for the agent

    Attributes:
        dataset (Dataset): Dataset used by the agent, when learning data
        reachStrategies (Strategy[]):
        iteration (int):
        lastIterationTime (int):
        iterationTimes (float[]):
        environment (Environment)
        options (dict)
        performer (Performer)
    """

    DISCRETE_STATES = False
    DISCRETE_ACTIONS = False

    PLANNER_CLASS = Planner
    PERFORMER_CLASS = Performer

    def __init__(self, host, dataset=None, performer=None, planner=None, options={}):
        super().__init__('Agent', host.spaceManager)
        manage(self).attach_counter(AsyncCounter(self))
        self.logger.tag = 'agent'

        self.host = host
        self.host.hosting = self

        self.environment = host.world.spaceManager
        self.assertDiscrete(self.environment)

        self.dataset = dataset
        self.options = options
        self.scheduled = None
        self.iterationEvent = threading.Event()

        # self.iteration = 0
        self.episode = 0
        self.iterationByEpisode = []
        self.iterationType = {}

        self.lastIterationEpisode = 0

        self.testStrategies = StrategySet(agent=self)

        self.performer = parameter(performer, self.PERFORMER_CLASS(self, options=options))
        self.planner = parameter(planner, self.PLANNER_CLASS(self, options=options))

        # debug metrics
        self.lastIterationTime = -1
        self.iterationTimes = []

    def __repr__(self):
        return f'Agent {self.__class__.__name__}'

    def _serialize(self, serializer):
        """Returns a dictionary with serialized information.

        Args:
            options (dict): Parameters of the serialization

        Returns:
            type: Serialized Object
        """
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(
            self, ['options', 'testStrategies'], exportPathType=True))
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, environment, options={}, obj=None):
    #     """Deserialized a dict into an object.

    #     Args:
    #         dict_ (dict): Object data
    #         environment (Environment): The environment within which the agent will live and operate
    #         options (dict): Deserializing parameters
    #         obj (Agent): When subclassing, the object created by the deserialize method of the subclass
    #     """
    #     from ..utils.loaders import DataManager

    #     obj = obj if obj else cls(
    #         environment, options=dict_.get('options', {}))
    #     obj = Module._deserialize(dict_, options=options, obj=obj)

    #     for strategy in dict_.get('testStrategies', []):
    #         obj.testStrategies.add(DataManager.loadType(strategy['path'], strategy['type'])
    #                                .deserialize(strategy, obj, options=options))
    #     return obj

    @property
    def iteration(self):
        return manage(self).counter.t

    @property
    def counter(self):
        return manage(self).counter
    
    def syncCounter(self):
        self.counter.sync()

    def trainable(self):
        return False

    def testable(self):
        return True

    @property
    def discreteStates(self):
        return self.DISCRETE_STATES

    @property
    def discreteActions(self):
        return self.DISCRETE_ACTIONS

    def assertDiscrete(self, environment, autochange=True):
        if self.discreteStates and not environment.discreteStates or self.discreteActions and not environment.discreteActions:
            if not environment.CAN_BE_DISCRETIZED:
                autochange = False
            if not autochange:
                raise Exception('Trying to apply a discrete learner to a continuous environment!\n' +
                                'Check if you have forgotten to call env.discretizeStates/Actions = True?')
            if self.discreteStates and not environment.discreteStates:
                environment.discretizeStates = True
            if self.discreteActions and not environment.discreteActions:
                environment.discretizeActions = True
    
    def actions(self, onlyPrimitives=True):
        return self.host.actions()
    
    def observe(self, formatParameters=None):
        return self.host.observeFrom(formatParameters=formatParameters)
    
    def schedule(self, method, *args, **kwargs):
        if self.environment.threading:
            def func():
                method(*args, **kwargs)
                self.scheduled = None

            self.scheduled = func
        else:
            method(*args, **kwargs)

    def reach(self, configOrGoal=MoveConfig()):
        if not isinstance(configOrGoal, MoveConfig):
            configOrGoal = MoveConfig(goal=configOrGoal)
        self.test(configOrGoal)

    def test(self, config=MoveConfig()):
        if config.goal and self.dataset:
            config.goal = self.dataset.convertData(config.goal)
        config.exploitation = True

        self.logger.debug2(f'Testing {config}')

        self.syncCounter()
        self.iterationType[self.iteration] = 'evaluation' if config.evaluating else 'test'
        self.schedule(self._test, config)

    def _test(self, config):
        self.syncCounter()
        memory = self.testStrategies.sample().run(config)
        self.syncCounter()
        self.iterationType[self.iteration] = 'end'

        self._postTest(memory, config)
    
    def _postTest(self, memory, config):
        pass

    def perform(self, action):
        self.performer.performActions(action)

    def _performAction(self, action, config=MoveConfig()):
        self.host.scheduledAction = True
        self.environment.execute(action, config=config, agent=self, sync=True)

    def step(self, action, countIteration=True):
        result = self.environment.step(action)
        if countIteration:
            self.iteration += 1
        return result
    
    def propertySpace(self, filter_=None, kind=None):
        space = self.environment.world.cascadingProperty(filter_).space
        if self.dataset:
            space = space.convertTo(spaceManager=self.dataset, kind=kind)
        return space
    
    def getIterationType(self, iteration):
        last = None
        for i in self.iterationType:
            if i > iteration:
                if last is None or self.iterationType[last] == 'end':
                    return ''
                return self.iterationType[last]
            last = i
        return ''

    # def addReachStrategy(self, strategy):
    #     """Add a Strategy designed to perform a task in a certain way.
    #
    #     Args:
    #         strategy (Strategy): The given Strategy, should possesses a reach method
    #     """
    #     if strategy in self.reachStrategies:
    #         return
    #     assert strategy.agent == self
    #     self.reachStrategies.append(strategy)
    #     self.addChildModule(strategy)
    #
    # def reach(self, goalOrConfig, config=MoveConfig()):
    #     """Tries to reach a given goal with constraints.
    #
    #     Args:
    #         goalOrConfig (Goal | MoveConfig): The Goal to reach (or a MoveConfig describing the Goal to reach)
    #         config (MoveConfig): Configuration for executing the task
    #     """
    #     if isinstance(goalOrConfig, MoveConfig):
    #         config = goalOrConfig
    #     else:
    #         config.goal = goalOrConfig
    #     if not self.reachStrategies:
    #         raise Exception('{} has no strategy for reaching a goal'.format(self))
    #     if self.dataset:
    #         config.goal = config.goal.convertTo(self.dataset)
    #     config.exploitation = True
    #     return self.reachStrategies[0].reach(config)
    #
    # def measureIterationTime(self):
    #     """Measures the execution time at each iteration
    #     """
    #     currentTime = int(round(time.time() * 1000))
    #     if self.lastIterationTime > 0:
    #         self.iterationTimes.append((self.iteration, currentTime - self.lastIterationTime))
    #     self.lastIterationTime = currentTime
