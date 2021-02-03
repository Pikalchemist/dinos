'''
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import numpy as np

from exlab.interface.serializer import Serializable, Serializer

from dino.data.data import SingleObservation
from dino.data.space import Space


class UnboundedProperty(Exception):
    """
    Short summary.
    """
    pass


class Property(Serializable):
    """
    An abstract class corresponding to features and effectors
    """

    def __init__(self, entity, name, dim=None, absoluteName=None, visual=False):
        self.name = name
        self.absoluteName = absoluteName
        if self.absoluteName and entity.findAbsoluteName(self.absoluteName):
            raise Exception(f'An entity/property named {self.absoluteName} already exists within {entity.root.reference()}! \
                             Names should be uniques.')
        self.entity = entity
        self.entity.addProperty(self)

        self._dim = dim
        self.space = None
        self.discretization = 1.
        self.actions = {}

        self.learnable = True
        self.visual = visual

        if entity.activated:
            self._activate()

    def _activate(self):
        if self.space:
            return
        if not self._dim and self.observable():
            self._dim = len(self.observePlain())
        if self._dim is None:
            raise Exception(
                f'You should specify the dimension of the effector parameter space for {self}')
        self.space = Space(self.entity.manager, self._dim)
        self.space._property = self

    def _sid(self, serializer, raw=False):
        if self.absoluteName:
            return self.absoluteName
        return serializer.uid('property', f'{self.entity._sid(serializer, raw=True)}.{self.name}')

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['name', 'absoluteName', 'entity', 'dim'])
        return dict_

    @property
    def dim(self):
        return self._dim

    @property
    def discretizeStates(self):
        return self.entity.discretizeStates

    @property
    def discretizeActions(self):
        return self.entity.discretizeActions

    def observable(self):
        return False

    def controllable(self):
        return False

    @property
    def bounded(self):
        return self.space is not None

    # def bindSpace(self, spaces):
    #     fullname = self.fullname()
    #     if fullname[0] == '.':
    #         fullname = fullname[1:]
    #     spaces = [space for space in spaces if space.effector == fullname or space.observable == fullname]
    #     if spaces:
    #         self.space = spaces[0]

    # points
    def point(self, value, relative=None):
        return self.space.point(value, relative=relative)

    def goal(self, value, relative=None):
        return self.space.goal(value, relative=relative)

    def action(self, value, relative=None):
        return self.space.action(value, relative=relative)

    def zero(self, relative=None):
        return self.space.zero(relative=relative)

    def plainZero(self):
        return self.space.plainZero()

    def fullname(self):
        return self.entity.fullname() + "." + self.name

    def reference(self):
        if self.absoluteName:
            return f'#{self.absoluteName}'
        return f'{self.entity.reference(short=True)}.{self.name}'

    def icon(self):
        return '👁' if self.observable() else '' + '🕹' if self.controllable() else ''

    def __repr__(self):
        absName = f'#{self.absoluteName}' if self.absoluteName else ''
        return f"{self.icon()}{absName}'{self.entity.reference(short=True)}.{self.name}'"


class Effector(Property):
    """
    An actionable element
    """

    def __init__(self, entity, name, dim=1, absoluteName=None, actions={}):
        Property.__init__(self, entity, name, dim=dim,
                          absoluteName=absoluteName)
        self.learnable = False
        self.actions = {key: list(param) for key, param in actions.items()}

    def controllable(self):
        return True

    def performPlain(self, parameters):
        if not isinstance(parameters, list):
            parameters = [parameters]
        if len(parameters) != self._dim:
            raise Exception((f'Parameters size for {self} mismatch: expecting {self._dim} and received {len(parameters)} ' +
                             '(did you forget to specify the dim in the constructor?)'))
        self._performPlain(parameters)

    def _performPlain(self, parameters):
        pass

    def discrete(self):
        return self.entity.discreteActions

    def perform(self, action):
        if self.space is None:
            raise UnboundedProperty(
                f"{self.name} of {self.entity} is not bounded to any space")
        parameters = action.projection(self.space)
        self.performPlain(parameters.valueOrdered)


class Observable(Property):
    """
    A readable property
    """

    def __init__(self, entity, name, absoluteName=None, discretization=1., visual=False):
        Property.__init__(self, entity, name,
                          absoluteName=absoluteName, visual=visual)
        self.converter = None
        self.visualizer = None
        self.saver = "save"
        if discretization:
            self.discretization = discretization

    def observable(self):
        return True

    def observePlain(self):
        obs = self._observePlain()
        if not isinstance(obs, np.ndarray):
            if not isinstance(obs, list) and not isinstance(obs, tuple):
                obs = np.array([obs])
            else:
                obs = np.array(obs)
        if self.discretizeStates:
            obs = (obs // self.discretization).astype(np.int)
        return obs

    def discrete(self):
        return np.issubdtype(self.observePlain().dtype.type, np.integer)

    def _observePlain(self):
        return []

    def observe(self, formatParameters=None):
        if self.space is None:
            raise UnboundedProperty(
                f"{self.name} of {self.entity} is not bounded to any space")
        return SingleObservation(self.space, self.space.formatData(self.observePlain(),
                                                             formatParameters=formatParameters))

    @property
    def dim(self):
        assert len(self.observePlain()) == self._dim
        return self._dim

    def set_converter(self, converter_name):
        self.converter = converter_name

    def set_saver(self, saver_name):
        self.saver = saver_name

    def setVisualizer(self, visualizer):
        self.visualizer = visualizer

    def visualize(self, data=None, draw=True):
        if self.visualizer:
            self.visualizer(data=data, draw=draw)

    def convert_to_space(self, value):
        if self.converter:
            return getattr(self.entity.__class__, self.converter)(self.entity, value, to_feature=False)
        return value

    def convert_to_feature(self, value):
        if self.converter:
            return getattr(self.entity.__class__, self.converter)(self.entity, value, to_feature=True)
        return value

    def save(self):
        return getattr(self.entity.__class__, self.saver)(self.entity)


class MethodObservable(Observable):
    """
    A property provided by a given method
    """

    def __init__(self, entity, name, method, absoluteName=None, discretization=None, visual=False):
        self.method = method
        Observable.__init__(
            self, entity, name, absoluteName=absoluteName, discretization=discretization, visual=visual)

    def _observePlain(self):
        return self.method(property=self)


class FunctionObservable(Observable):
    """
    A property provided by a given function
    """

    def __init__(self, entity, name, function, absoluteName=None, discretization=None, visual=False, proxySpace=False):
        self.function = function

        if proxySpace:
            existingProperty = entity.propertyItem(name)

        Observable.__init__(
            self, entity, name, absoluteName=absoluteName, discretization=discretization, visual=visual)

        if proxySpace and existingProperty:
            self.space = existingProperty.space

    def _observePlain(self):
        return self.function(entity=self.entity, property=self)


class AttributeObservable(Observable):
    """
    A property directly read from an object attribute
    """

    def __init__(self, entity, name, attributeName, absoluteName=None, discretization=None, visual=False):
        self.attributeName = attributeName
        Observable.__init__(
            self, entity, name, absoluteName=absoluteName, discretization=discretization, visual=visual)

    def _observePlain(self):
        return getattr(self.entity, self.attributeName)


class MethodEffector(Effector):
    """
    An effector provided by a given method
    """

    def __init__(self, entity, name, method, dim, absoluteName=None, actions={}):
        self.method = method
        Effector.__init__(self, entity, name, dim=dim,
                          absoluteName=absoluteName, actions=actions)

    def _performPlain(self, parameters):
        return self.method(parameters, property=self)


class FunctionEffector(Effector):
    """
    An effector provided by a given function
    """

    def __init__(self, entity, name, function, dim, absoluteName=None, actions={}):
        self.function = function
        Effector.__init__(self, entity, name, dim=dim,
                          absoluteName=absoluteName, actions=actions)

    def _performPlain(self, parameters):
        return self.function(parameters, entity=self.entity, property=self)
