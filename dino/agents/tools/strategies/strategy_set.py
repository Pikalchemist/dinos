from exlab.interface.serializer import Serializable
import random


class StrategySet(set, Serializable):
    def __init__(self, strategies=[], agent=None):
        super().__init__()
        self.agent = agent
        for strategy in strategies:
            self.add(strategy)

    def _serialize(self, serializer):
        return [serializer.serialize(item) for item in self]

    def add(self, strategy):
        if strategy in self:
            return
        if self.agent:
            assert strategy.agent == self.agent
            # self.agent.addChildModule(strategy)
        set.add(self, strategy)

    def sample(self):
        if not self:
            raise Exception('No strategy available! Please add one first')
        return random.choice(list(self))
