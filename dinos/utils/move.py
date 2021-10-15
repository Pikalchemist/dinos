'''
    File name: objects.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import copy
import random

from exlab.interface.serializer import Serializable
from exlab.utils.io import parameter

from dinos.agents.tools.planners.planner import PlanSettings

from .result import Result


"""
Misc objects
"""


class MoveConfig(Serializable):
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
                 changeContext=False, lastEvent=None, sampling='', iterations=1, iteration=-1, allowReplanning=True,
                 evaluating=False, plannerSettings=None):
        self.exploitation = exploitation
        self.evaluating = evaluating
        self.depth = depth

        self.strategy = strategy
        self.model = model
        self.goal = goal
        self.goalContext = goalContext
        self.changeContext = changeContext
        self.absoluteGoal = None
        self.sampling = sampling

        self.allowReplanning = allowReplanning
        self.plannerSettings = parameter(plannerSettings, PlanSettings())

        self.iterations = iterations
        self.iteration = iteration

        self.result = Result(self)
        self.plannerSettings.result = self.result
    
    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['exploitation', 'evaluating', 'depth', 'goal', 'goalContext', 'absoluteGoal',
                                            'sampling', 'allowReplanning', 'iterations', 'iteration', 'result'],
                                     foreigns=['strategy', 'model'])
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls()
        return super()._deserialize(dict_, serializer, obj)
    
    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        for attr in ['exploitation', 'evaluating', 'depth', 'goal', 'goalContext', 'absoluteGoal',
                     'sampling', 'allowReplanning', 'iterations', 'iteration', 'strategy', 'model']:
            if attr in dict_:
                setattr(self, attr, serializer.deserialize(dict_.get(attr)))
        if 'result' in dict_:
            serializer.deserialize(dict_.get('result'), obj=self.result)

    def clone(self, **kwargs):
        new = copy.copy(self)
        for key, value in kwargs.items():
            setattr(new, key, value)
        
        new.result = new.result.clone(new)
        return new

    def nextdepth(self, **kwargs):
        kwargs['depth'] = self.depth + 1
        return self.clone(**kwargs)
    
    @property
    def training(self):
        return not self.evaluating and not self.exploitation

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
