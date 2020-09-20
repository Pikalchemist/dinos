'''
    File name: objects.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import copy
import random


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
                 lastEvent=None, sampling="", iterations=1, iteration=-1, allowReplanning=True, evaluating=False):
        self.model = model
        self.exploitation = exploitation
        self.depth = depth
        self.strategy = strategy
        self.goal = goal
        self.goalContext = goalContext
        self.lastEvent = lastEvent
        self.sampling = sampling
        self.iterations = iterations
        self.iteration = iteration
        self.allowReplanning = allowReplanning
        self.evaluating = evaluating

    def clone(self, **kwargs):
        new = copy.copy(self)
        for key, value in kwargs.items():
            setattr(new, key, value)
        return new

    def nextdepth(self, **kwargs):
        kwargs['depth'] = self.depth + 1
        return self.clone(**kwargs)

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
