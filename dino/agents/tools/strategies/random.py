import copy
import random
import numpy as np

# from ...utils.serializer import getReference
# from ...data.dataset import Action
from dino.data.data import Action, SingleAction, ActionList
# from dino.data.space import *
from dino.utils.move import MoveConfig

from .strategy import Strategy


class RandomStrategy(Strategy):
    """The random exploration of actions strategy."""

    def __init__(self, agent, name=None, performer=None, planner=None, options={}):
        super().__init__(agent, name=name, planner=planner, performer=performer,
                         options=options)
        self.exploreNonPrimitive = self.options.get(
            'exploreNonPrimitive', False)

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(self, exportPathType=True))
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, agent, options=None, obj=None):
    #     obj = obj if obj else cls(agent, dict_.get(
    #         'name'), options=dict_.get('options', {}))
    #     obj = Strategy._deserialize(dict_, agent, options=options, obj=obj)
    #     return obj

    def _runIteration(self, config):
        # Test a random action
        self.testRandomAction(config)

    # test modifications on both simple and complex agents
    def testRandomAction(self, config=MoveConfig(), actionSpaces=None, zero=False):
        """Build and test a random action."""
        if not actionSpaces:
            actionSpaces = [a.space for a in self.agent.actions(onlyPrimitives=not self.exploreNonPrimitive)]
        if self.agent.dataset:
            actionSpaces = self.agent.dataset.controllableSpaces(actionSpaces)
        space = random.choice(actionSpaces)

        # self.logger.debug("Random action performed in {} selected among {} spaces"
        #                   .format(getReference(space), len(actionSpaces)), 'BABLG')
        '''#space = self.env.dataset.actionExplorationSpaces[0]
        n = 0
        actions = []
        # if self.complex_actions <= 0:
        cost = 1.0
        r = random.random()
        # Keep chaining current actions with primitives to build a complex action
        while len(actions) == 0 or r < cost:
            action = SingleAction(space, [random.uniform(minb, maxb) for minb, maxb in space.bounds])
            actions.append(action)
            n += 1
            r = random.random()
            # dev
            cost = 0.
            #cost = 1.0 / (self.k ** n)
        # else:
        #     n = self.complex_actions
        #     for i in range(self.complex_actions):
        #         action = SingleAction(space, [random.uniform(minb, maxb) for minb, maxb in space.bounds])
        #         actions.append(action)'''

        actions = []
        if zero:
            action = space.zero()
        else:
            action = SingleAction(space, [random.uniform(minb, maxb) for minb, maxb in space.bounds])
        actions.append(action)

        actions = ActionList(*actions)
        self.testActions(actions, config)
