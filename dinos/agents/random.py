from .agent import Agent
from dinos.agents.tools.strategies.random import RandomStrategy


class RandomAgent(Agent):
    def __init__(self, host, dataset=None, performer=None, planner=None, options={}):
        super().__init__(host, dataset=dataset, performer=performer,
                         planner=planner, options=options)
        self.testStrategies.add(RandomStrategy(self))
