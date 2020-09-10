from .agent import Agent
from dino.agents.tools.strategies.random import RandomStrategy


class RandomAgent(Agent):
    def __init__(self, host, performer=None, options={}):
        super().__init__(host, performer, options)
        self.testStrategies.add(RandomStrategy(self))
