import copy
import random
import numpy as np

from exlab.modular.module import Module

from exlab.utils.io import parameter
from dino.utils.move import MoveConfig
from dino.utils.maths import linearValue

# from ...data.dataset import Action
# from dino.data.data import *
from dino.data.space import SpaceKind
from dino.data.event import InteractionEvent
from dino.data.path import ActionNotFound



class Strategy(Module):
    """Strategy usable by a learning agent."""

    def __init__(self, agent, name=None, performer=None, planner=None, options={}):
        """
        name string: name of the strategy
        """
        self.name = name if name else (self.__class__.__name__[
                                       :1].lower() + self.__class__.__name__[1:])
        super().__init__(f'Strategy {self.name}', agent, loggerTag='strategy')

        self.agent = agent
        self.options = options
        # self.iterations = options.get('iterations', 1)

        # Short term memory containing all actions & outcomes reached during the last learning episode
        self.memory = []
        # self.config = None
        # self.n = 0

        # self.complex_actions = complex_actions
        # self.resetEnv = True

        self.performer = parameter(performer, self.agent.performer)
        self.planner = parameter(planner, self.agent.planner)
        # self.addChildModule(self.performer)

    def __repr__(self):
        return f'Strategy {self.name}'
    
    def _sid(self, serializer):
        return serializer.uid('strategy', self.name)

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(self, ['name']))
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
        # self.config = config
        # self.n = 0

    def _run(self, config):
        self.runIteration(config)

    def _postRun(self, config):
        return self.memory
    
    # def processMemory(self, memory=None):
    #     memory = parameter(memory, self.memory)
    #     self.agent.addMemory(memory, self.config)

    def runIteration(self, config):
        self.agent.environment.checkTerminated()
        self.agent.syncCounter()
        self.agent.environment.setupIteration(config)
        self._runIteration(config)

    def _runIteration(self, config):
        pass

    def reachGoalContext(self, config):
        if not config.goalContext:
            return True

        probFirstPass = linearValue(1., 0., (config.goalContext.space.number - 20) / 100)
        if not config.exploitation and random.uniform(0, 1) < probFirstPass:
            return True

        settings = config.plannerSettings.clone(context=True)
        if config.goal and not config.goal.space.matches(config.goalContext.space, kindSensitive=False):
            settings.dontMoveSpaces.append(config.goal.space)
        try:
            path, _, _ = self.planner.planDistance(config.goalContext.setRelative(False), settings=settings)
        except ActionNotFound:
            config.result.reachedContext = 'planning failed'
            path = None

        if not path:
            self.logger.warning(
                f"Planning failed for pre-goal context {config.goalContext}")
            return False
        
        self.testPath(path, config.clone(model=None))

        config.result.reachedContext = self.agent.environment.state().context().projection(config.goalContext.space)
        return True

        # config.goalContext = config.goalContext.convertTo(
        #     self.agent.dataset, kind=SpaceKind.BASIC)

        # try:
        #     paths, _ = self.planner.planDistance(config.goalContext)
        #     self.logger.debug("Planning generated for context goal {} in {} steps"
        #                       .format(config.goalContext, len(paths)), 'PLAN')
        #     self.testGoal(config.goalContext, paths,
        #                   config.clone(model=None))
        # except ActionNotFound:
        #     self.logger.warning("Context planning failed for goal {}, switching to random"
        #                         .format(config.goalContext), 'PLAN')
        #     return False

    def testActions(self, actions, config=MoveConfig()):
        try:
            self.testPath(self.planner.planActions(actions), config)
        except ActionNotFound:
            return

    # def testGoal(self, goal, paths=None, config=MoveConfig()):
    #     results = self.performer.performGoal(goal, paths, config)
    #     # print('Tested', results)
    #     # self.n += InteractionEvent.incrementList(results, self.n)
    #     self.memory += results

    def testPath(self, path, config=MoveConfig()):
        """Test a specific complex action and store consequences in memory."""
        results = self.performer.perform(path, config)
        # self.n += InteractionEvent.incrementList(results, self.n)
        self.memory += results
        # self.n += 1

    def __deepcopy__(self, a):
        newone = type(self).__new__(type(self))
        newone.__dict__.update(self.__dict__)
        newone.parent = None
        newone.modules = []
        newone.agent = None
        newone.__dict__ = copy.deepcopy(newone.__dict__)
        return newone
