'''
    File name: planner.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import numpy as np
from scipy.spatial.distance import euclidean

import itertools
import random
import math

from .data import Data, Goal, SingleAction, Action, ActionList

"""
Paths[
    Path[
        PathNode(Goal, Action)
    , ...]
, ...]
"""


class ActionNotFound(Exception):
    def __init__(self, message, minDistanceReached=None):
        super().__init__(message)
        self.minDistanceReached = minDistanceReached


class Paths(object):
    """
    Represents multiple Path that should be executed simultaneously
    """

    def __init__(self, paths=[]):
        self.__paths = paths

    @property
    def paths(self):
        return self.__paths

    @paths.setter
    def paths(self, paths):
        self.__paths = paths

    def length(self):
        return np.sum([path.length() for path in self.__paths])

    def extends(self, paths):
        assert len(self) == 1
        assert len(paths) == 1
        self.__paths[0].extends(paths.__paths[0])
        return self

    def getGroupedActionList(self):
        if len(self) > 1:
            return [(None, self.getActionList())]
        return self.__paths[0].getGroupedActionList()

    def getActionList(self):
        actions = itertools.zip_longest(
            *[path.getActionList() for path in self.__paths])
        actions = [Action(*[_f for _f in tup if _f]) for tup in actions]
        actions_ = []
        for a in actions:
            suba = itertools.zip_longest(
                *[s if isinstance(s, ActionList) else [s] for s in a])
            suba = [Action(*[_f for _f in tup if _f]) for tup in suba]
            suba_ = []
            for s in suba:
                s_ = []
                for s2 in s:
                    if isinstance(s2, ActionList):
                        s_ += s2.get()
                    else:
                        s_.append(s2)
                suba_.append(Action(*s_))
            actions_ += suba_
        #print("{} --> {}".format(actions, actions_))
        return ActionList(*actions_)

    def __iter__(self):
        return self.__paths.__iter__()

    def __len__(self):
        return len(self.__paths)

    def __getitem__(self, key):
        return self.__paths[key]

    def toStr(self, short=False):
        parts = ' | '.join([path.toStr(short=True) for path in self.__paths])
        return f'Paths({parts})'

    def __repr__(self):
        return self.toStr()


class Path(object):
    """
    Represents multiple goal nodes to be executed in order
    """

    def __init__(self, nodes):
        self.__nodes = nodes

    def extends(self, path):
        self.__nodes += path.__nodes

    def nodes(self):
        return self.__nodes

    def getGroupedActionList(self):
        return [node.getGroupedActionList() for node in self]

    def getActionList(self):
        return [action for node in self for action in node.getActionList()]

    def length(self):
        return np.sum([node.length() for node in self.__nodes])

    def __len__(self):
        return len(self.__nodes)

    def __iter__(self):
        return self.__nodes.__iter__()

    def __add__(self, other):
        return self.__class__(self.__nodes + other.__nodes)

    def __getitem__(self, key):
        return self.__nodes[key]

    def toStr(self, short=False):
        parts = '\n-> '.join([node.toStr(short=True) for node in self.__nodes])
        return f'<<\n   {parts}\n>>'

    def __repr__(self):
        return self.toStr()


class PathNode(Paths):
    """
    Represents a couple (action, goal) where the action should reach the goal
    """

    def __init__(self, action=None, goal=None, model=None, pos=None, parent=None, state=None):
        super().__init__()
        self.action = action
        self.goal = goal
        self.model = model
        self.pos = pos
        self.parent = parent
        self.state = state
        self.valid = False

    def length(self):
        if self.paths:
            return Paths.length(self)
        else:
            return self.goal.length()

    def getGroupedActionList(self):
        return self, self.getActionList()
        # action = self.getActionList()
        # return (self, action if isinstance(action, ActionList) else ActionList(action))

    def getActionList(self):
        if not self.paths:
            return ActionList(self.action)
        else:
            return super().getActionList()

    def toStr(self, short=False):
        paths = super().toStr(True) if self.paths else ''
        return f"{'' if short else 'Node'} {type(self.action)}({self.action.toStr(short=True) if self.action else ''} - to reach→ {self.goal.toStr(short=True) if self.goal else 'NoGoal'}){paths}"
        # return "Node: {} ({}, {}) {} [{}]".format(self.goal, self.action, self.model, paths, self.state)

    def __repr__(self):
        return self.toStr()
