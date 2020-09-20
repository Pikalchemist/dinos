import copy
import random
import numpy as np

from exlab.modular.module import Module

from exlab.utils.io import parameter
from dino.utils.move import MoveConfig

# from ...data.dataset import Action
# from dino.data.data import *
from dino.data.space import SpaceKind
from dino.data.event import InteractionEvent
from dino.data.path import ActionNotFoundException



class Strategy(Module):
    """Strategy usable by a learning agent."""

    def __init__(self, agent, name=None, performer=None, planner=None, options={}):
        """
        name string: name of the strategy
        """
        self.name = name if name else (self.__class__.__name__[
                                       :1].lower() + self.__class__.__name__[1:])
        super().__init__(f'Strategy {self.name}', agent)

        self.agent = agent
        self.options = options
        # self.iterations = options.get('iterations', 1)

        # Short term memory containing all actions & outcomes reached during the last learning episode
        self.memory = []
        # self.n = 0

        # self.complex_actions = complex_actions
        # self.resetEnv = True

        self.performer = parameter(performer, self.agent.performer)
        self.planner = parameter(planner, self.agent.planner)
        # self.addChildModule(self.performer)

    def __repr__(self):
        return f'Strategy {self.name}'

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(self, ['name'], exportPathType=True))
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, agent, options=None, obj=None):
    #     obj = obj if obj else cls(agent, dict_.get(
    #         'name'), options=dict_.get('options', {}))
    #     obj = Module._deserialize(dict_, options=options, obj=obj)
    #     return obj

    def available(self, space):
        """Says if the strategy is available to the agent."""
        return True

    def trainable(self):
        return True

    def testable(self):
        return True

    def run(self, config):
        """Runs the strategy in train or test mode (function used when the strategy does not require a goal/task)."""
        self._preRun(config)
        while len(self.memory) < config.iterations:
            self._run(config)
        return self._postRun(config)

    def _preRun(self, config):
        self.agent.syncCounter()
        self.memory = []
        # self.n = 0

    def _run(self, config):
        self.runIteration(config)

    def _postRun(self, config):
        memory = self.memory
        if self.agent.dataset:
            for event in memory:
                event.convertTo(spaceManager=self.agent.dataset, toData=True)
        return memory

    def runIteration(self, config):
        self.agent.syncCounter()
        self.agent.environment.setupIteration(config)
        self._runIteration(config)

    def _runIteration(self, config):
        pass

    # def reachGoalContext(self, config):
    #     if config.goalContext:
    #         print('Reaching goal context...')
    #         config.goalContext = config.goalContext.convertTo(
    #             self.agent.dataset, kind=SpaceKind.BASIC)

    #         try:
    #             paths, _ = self.planner.planDistance(config.goalContext)
    #             self.logger.debug("Planning generated for context goal {} in {} steps"
    #                               .format(config.goalContext, len(paths)), 'PLAN')
    #             self.testGoal(config.goalContext, paths,
    #                           config.clone(model=None))
    #         except ActionNotFoundException:
    #             self.logger.warning("Context planning failed for goal {}, switching to random"
    #                                 .format(config.goalContext), 'PLAN')
    #             return False

    def testActions(self, actions, config=MoveConfig()):
        try:
            self.testPaths(self.planner.planActions(actions), config)
        except ActionNotFoundException:
            return

    def testGoal(self, goal, paths=None, config=MoveConfig()):
        results = self.performer.performGoal(goal, paths, config)
        # print('Tested', results)
        # self.n += InteractionEvent.incrementList(results, self.n)
        self.memory += results

    def testPaths(self, paths, config=MoveConfig()):
        """Test a specific complex action and store consequences in memory."""
        results = self.performer.perform(paths)
        # self.n += InteractionEvent.incrementList(results, self.n)
        self.memory += results

    def __deepcopy__(self, a):
        newone = type(self).__new__(type(self))
        newone.__dict__.update(self.__dict__)
        newone.parent = None
        newone.modules = []
        newone.agent = None
        newone.__dict__ = copy.deepcopy(newone.__dict__)
        return newone
