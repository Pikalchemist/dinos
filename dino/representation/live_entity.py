'''
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import numpy as np

from exlab.interface.serializer import Serializable, Serializer
from exlab.utils.text import colorText, Colors
from exlab.utils.ensemble import Ensemble

from dino.data.data import Observation
from dino.data.space import SpaceKind

from .entity import Entity


"""

Scene[
    Entity(
        [Observable, ...]
        [Effector, ...]
    ),
...]

"""


class LiveEntity(Entity):
    """Represents a world entity with a set of properties (features and effectors)

    Args:
        kind (string): the type of our entity
        absoluteName (string): Absolute name used to point to this entity. Unique

    Attributes:
        index (int): Absolute index
        indexKind (int): Index for the given entity type
        parent (Entity):
        activated (bool):
        absoluteName
        kind

    """
    PHYSICAL = False
    DISCRETE_STATES = False
    DISCRETE_ACTIONS = False
    CAN_BE_DISCRETIZED = False

    def __init__(self, kind, absoluteName='', disconnected=False, manager=None):
        super().__init__(kind, absoluteName, disconnected, manager=manager)
        self.host = False
        self.hosting = None
        self.scheduledAction = False

        self.physicals = []
        self.actionQueue = []

    # def cid(self):
    #     return self.index

    # def gid(self):
    #     if self.absoluteName:
    #         return Serializer.make_gid(self, self.kind, self.absoluteName)
    #     return Serializer.make_gid(self, self.kind, self.indexKind)

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['kind', 'absoluteName', 'index', 'indexKind', 'parent', '_children', '_properties'])
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, options=None, obj=None):
    #     obj = obj if obj else cls(dict_.get('kind'), dict_.get(
    #         'absoluteName', ''), disconnected=True)
    #     return obj

    def markAsHost(self):
        self.host = True
    
    def hosts(self):
        children = self.cascadingChildren()
        return [host for host in children if host.host]
    
    def findHost(self, name=''):
        if not name:
            return next(iter(self.hosts()), None)
        return next((host for host in self.hosts() if host.absoluteName == name), None)

    def _update(self, dt):
        pass

    def draw(self, *args, **kwargs):
        pass

    def update(self, dt=0.):
        self._update(dt)
        for action in list(self.actionQueue):
            if action.update(dt):
                self.actionQueue.remove(action)
        for child in self._children:
            child.update(dt)

    def performAction(self, function, args=(), duration=0., step=None):
        self.actionQueue.append(ActionExecution(
            function, args, duration, step))


class ActionExecution(object):
    TOLERANCE_STEP = 1e-5

    def __init__(self, function, args=(), duration=0., step=None):
        self.function = function
        self.args = args
        self.duration = duration
        self.step = step
        self.elapsed = 0.
        self.lastExecution = -1000.

    def update(self, dt):
        if not self.step or self.elapsed - self.lastExecution >= self.step - self.TOLERANCE_STEP:
            self._perform()
            self.lastExecution = self.elapsed

        self.elapsed += dt
        if self.elapsed >= self.duration:
            return True
        return False

    def _perform(self):
        self.function(*self.args)
