from .model import ModelLearner
from dinos.agents.tools.strategies.random import RandomStrategy


class RandomLearner(ModelLearner):
    def __init__(self, environment, dataset=None, performer=None, planner=None, options={}):
        super().__init__(environment=environment, dataset=dataset,
                         performer=performer, planner=planner, options=options)
        self.trainStrategies.add(RandomStrategy(self))
