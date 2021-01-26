import numpy as np
import random
import copy
import math
import time
import threading

from exlab.lab.counter import AsyncCounter
from exlab.modular.module import Module, manage
from exlab.interface.serializer import Serializer

from exlab.utils.io import parameter
from dino.utils.move import MoveConfig
from dino.utils.database import Database

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
        super().__init__('Agent', host.manager)
        manage(self).attach_counter(AsyncCounter(self))
        self.logger.tag = 'agent'

        self.host = host
        self.host.hosting = self

        self.environment = host.world.manager
        self.assertDiscrete(self.environment)

        self.dataset = dataset
        self.options = options
        self.scheduled = None
        self.iterationEvent = threading.Event()

        self.configs = {}
        self.analyse = self.__class__.ResultAnalyser(self)

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
            self, ['options', 'testStrategies', 'iteration', 'configs'], foreigns=['host']))
        return dict_
    
    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls(serializer.find(dict_.get('host')),
                      serializer.deserialize(dict_.get('dataset')),
                      options=dict_.get('options', {}))
        
        return super()._deserialize(dict_, serializer, obj)
    
    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        env = serializer.get('environment')
        if env.iteration < dict_.get('iteration', 0):
            env.counter.iteration = dict_.get('iteration')
        self.syncCounter()

        self.configs = serializer.deserialize(dict_.get('configs'), default={})
    
    def deserializer(self):
        d = self.environment.deserializer()
        d.set('agent', self)
        if self.dataset:
            d.set('dataset', self.dataset)
            d.set('dataset', self.dataset, category='spaceManager')
        d.attach_finder('strategy', self.findStrategy)
        return d

    def findStrategy(self, name):
        return None

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

    def currentConfig(self, name=''):
        d = {
            'environment': self.environment.__class__.__name__,
            'learner': self.__class__.__name__,
            'name': name,
        }
        return d

    def save(self, name):
        serializer = Serializer()
        data = {'agent': self.serialize(serializer)}
        evaluator = self.environment.evaluator(self)
        if evaluator:
            data['evaluator'] = evaluator.serialize(serializer)
        db = Database.from_data(self.currentConfig(name), data)
        db.save()
        return db
    
    def load(self, path):
        db = Database.from_file(Database.databasedir / path)
        db.load()
        d = self.deserializer()
        if self.dataset:
            d.deserialize(db.data.get('agent', {}).get('dataset', {}), obj=self.dataset)
        self._postDeserialize(db.data.get('agent', {}), d)

        d.deserialize(db.data.get('evaluator', {}), obj=self.environment.evaluator(self, create=True))
    
    @classmethod
    def loadAgent(self, environment, path):
        db = Database.from_file(Database.databasedir / path / Database.FILENAME)
        db.load()
        d = environment.deserializer()
        agent = d.deserialize(db.data.get('agent', {}))
        d.deserialize(db.data.get('evaluator', {}), obj=environment.evaluator(agent, create=True))
        return agent

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
    
    def actions(self):
        return self.host.actions()

    def explorableSpaces(self, onlyPrimitives=False):
        spaces = [a.space for a in self.host.actions()]
        if not onlyPrimitives and self.dataset:
            spaces += self.dataset.controllableSpaces(explorable=True)
        return list(set(spaces))
    
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
        strategy = self.testStrategies.sample()
        config.strategy = strategy
        memory = strategy.run(config)
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
        return self.environment.propertySpace(filter_, kind=kind, dataset=self.dataset)
    
    def getIterationType(self, iteration):
        last = None
        for i in self.iterationType:
            if i > iteration:
                if last is None or self.iterationType[last] == 'end':
                    return ''
                return self.iterationType[last]
            last = i
        return ''

    class ResultAnalyser(object):
        def __init__(self, agent):
            self.agent = agent

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
