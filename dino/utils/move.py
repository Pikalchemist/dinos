'''
    File name: objects.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import copy
import random

from exlab.utils.io import parameter
from dino.agents.tools.planners.planner import PlanSettings

from .result import Result


"""
Misc objects
"""


class MoveConfig(object):
    """
    Represents a move configuration:
    - Exploitation:
        * Model
        * Goal
        * (recursion depth)
    - Training:
        - Goal oriented:
            * Model
            * Goal
            * Strategy
            * (recursion depth)
        - Action oriented:
            * Strategy
            * (recursion depth)
    """

    def __init__(self, model=None, exploitation=False, depth=0, strategy=None, goal=None, goalContext=None,
                 lastEvent=None, sampling='', iterations=1, iteration=-1, allowReplanning=True, evaluating=False,
                 plannerSettings=None):
        self.exploitation = exploitation
        self.evaluating = evaluating
        self.depth = depth

        self.strategy = strategy
        self.model = model
        self.goal = goal
        self.goalContext = goalContext
        self.absoluteGoal = None
        self.sampling = sampling

        self.allowReplanning = allowReplanning
        self.plannerSettings = parameter(plannerSettings, PlanSettings())

        self.iterations = iterations
        self.iteration = iteration

        self.result = Result(self)
        self.plannerSettings.result = self.result

    def clone(self, **kwargs):
        new = copy.copy(self)
        for key, value in kwargs.items():
            setattr(new, key, value)
        
        new.result = new.result.clone(new)
        return new

    def nextdepth(self, **kwargs):
        kwargs['depth'] = self.depth + 1
        return self.clone(**kwargs)

    def __repr__(self):
        if self.evaluating:
            prefix = "Evaluation"
            attrs = ['model', 'goal', 'absoluteGoal', 'depth']
        elif self.exploitation:
            prefix = "Exploit"
            attrs = ['model', 'goal', 'absoluteGoal', 'depth']
        elif self.goal:
            prefix = "Goal exploration"
            attrs = ['goal', 'absoluteGoal', 'goalContext',
                     'model', 'strategy', 'depth', 'sampling']
        else:
            prefix = "Action exploration"
            attrs = ['strategy', 'depth']
        params = ', '.join([f'{k}: {getattr(self, k)}' for k in attrs])
        return f'Config ({prefix}) [{params}] [{self.result}]'
