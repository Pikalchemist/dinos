'''
    File name: objects.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import copy
import random

from exlab.utils.io import parameter
from dino.agents.tools.planners.planner import PlanSettings


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
                 lastEvent=None, sampling="", iterations=1, iteration=-1, allowReplanning=True, evaluating=False,
                 plannerSettings=None):
        self.exploitation = exploitation
        self.evaluating = evaluating
        self.depth = depth

        self.strategy = strategy
        self.model = model
        self.goal = goal
        self.goalContext = goalContext

        self.reachedGoal = None
        self.reachedContext = None

        self.lastEvent = lastEvent
        self.sampling = sampling

        self.iterations = iterations
        self.iteration = iteration
        self.allowReplanning = allowReplanning

        self.plannerSettings = parameter(plannerSettings, PlanSettings())

    def clone(self, **kwargs):
        new = copy.copy(self)
        for key, value in kwargs.items():
            setattr(new, key, value)
        return new

    def nextdepth(self, **kwargs):
        kwargs['depth'] = self.depth + 1
        return self.clone(**kwargs)
    
    def results(self):
        score = 0.
        txt = ''

        if self.goal:
            txt += f'goal {self.goal} '
            if not self.reachedGoal:
                txt += f'not attempted'
            elif isinstance(self.reachedGoal, str):
                txt += f'{self.reachedGoal}'
            else:
                difference = (self.goal - self.reachedGoal).norm()
                score += difference / self.goal.space.maxDistance * 5.
                txt += f'and got {self.reachedGoal}, difference is {difference}'
            txt += f'|   '
        
        if self.goalContext:
            txt += f'context {self.goalContext} '
            if not self.reachedContext:
                txt += f'not attempted'
            elif isinstance(self.reachedContext, str):
                txt += f'{self.reachedContext}'
            else:
                difference = (self.goalContext - self.reachedContext).norm()
                score += difference / self.goalContext.space.maxDistance * 5.
                txt += f'and got {self.reachedContext}, difference is {difference}'
            txt += f'|   '

        valid = 'Ok' if score < 0.1 else 'Error'
        return f'{valid}: {score} ({txt})'

    def __repr__(self):
        if self.exploitation:
            prefix = "Exploit"
            attrs = ['model', 'goal', 'depth']
        elif self.goal:
            prefix = "Goal exploration"
            attrs = ['goal', 'goalContext', 'model', 'strategy', 'depth']
        else:
            prefix = "Action exploration"
            attrs = ['strategy', 'depth']
        params = ', '.join([k + ': ' + str(getattr(self, k)) for k in attrs])
        return f'Config ({prefix}) [{params}]'
