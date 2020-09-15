import sys
import copy
import math
import random
import numpy as np
from scipy.spatial.distance import euclidean

from exlab.interface.serializer import Serializable


class Data(Serializable):
    def __init__(self, *args):
        if len(args) == 0:
            # Null data
            self.space = None
            self.value = []
            self.valueOrdered = []
            self._parts = []
        else:
            if len(args) == 2 and not isinstance(args[0], Data):
                # Data(space, value)
                self.space = args[0]
                value = args[1].tolist() if isinstance(args[1], np.ndarray) else list(args[1])
                if len(self.space.cols) == 1:
                    self._parts = [SingleData(self.space, value)]
                else:
                    self._parts = []
                    for part in self.space:
                        self._parts.append(SingleData(part, value[:part.dim]))
                        del value[:part.dim]
            else:
                # Data(Data#0, Data#1, ...)
                self._parts = list(args)
                self.space = self._parts[0].space.spaceManager.multiColSpace(
                    [p.space for p in self._parts])

            self.abstract = any(p.abstract for p in self._parts)
            self.value = self.plain()
            self.valueOrdered = self.plainOrdered()

    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['space', 'value', 'valueOrdered'], exportPathType=True)
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, options={}, obj=None):
    #     obj = obj if obj else cls(dict_.get('iteration'), dict_.get('meanError'), dict_.get('meanStd'),
    #                               dict_.get('results'), dict_.get('models'))
    #     return obj

    # Operators
    def __iter__(self):
        return self._parts.__iter__()

    def __getitem__(self, arg):
        return self._parts[arg]

    def __len__(self):
        return len(self._parts)

    def __bool__(self):
        return len(self._parts) > 0

    def __neg__(self):
        return self.__class__(*(-i for i in self))

    def __mul__(self, value):
        return self.__class__(*(v*value for v in self))

    def __add__(self, other):
        return self.__class__(*(v + other.singleSpaceComposante(v.space) for v in self))

    def __sub__(self, other):
        return self.__class__(*(v - other.singleSpaceComposante(v.space) for v in self))

    def __eq__(self, other):
        if isinstance(other, Data):
            if self.space is None or other.space is None:
                return self.space == other.space
            return (self.space == other.space or self.space.native == other.space.native) and self.valueOrdered == other.valueOrdered
        return False
    
    def distanceTo(self, other):
        if not isinstance(other, Data) or not self.space.matches(other.space, kindSensitive=False):
            raise ValueError('Spaces should be the same for both data')

        return euclidean(self.valueOrdered, other.valueOrdered)

    def difference(self, previous):
        result = []
        for point in self.flat():
            if point.space.relative:
                result.append(
                    point - previous.singleSpaceComposante(point.space))
            else:
                result.append(point)
        return Observation(*result)

    def norm(self):
        return np.linalg.norm(self.value)

    def length(self):
        return np.linalg.norm(self.value)

    def setRelative(self, relative=None):
        if relative is not None:
            for part in self._parts:
                part.setRelative(relative)
        return self

    def absoluteData(self, state):
        return self.__class__(*[part.absoluteData(state) for part in self._parts])

    def relativeData(self, state):
        return self.__class__(*[part.relativeData(state) for part in self._parts])
    
    def clone(self):
        return self.__class__(*[part.clone() for part in self._parts])

    # Flatting
    def flat(self):
        return [x for p in self._parts for x in p.flat()]

    def plain(self):
        return [x for p in self._parts for x in p.plain()]

    def plainOrdered(self, spaceOrder=None, assertSize=True, fill=False):
        if not self._parts:
            return []
        if not spaceOrder:
            spaceOrder = self.space

        # parts = [x for s in spaceOrder.flatColsWithMultiRows for x in self.singleSpaceComposante(s).plainOrdered()]
        if self.abstract:
            parts = [x for s in spaceOrder.flatColsWithMultiRows for x in self.singleSpaceComposante(
                s).plainOrdered()]
        else:
            parts_ = [[list(map(self.singleSpaceComposante, possibility))
                       for possibility in group] for group in spaceOrder.groupedCols]
            parts = []
            for group in parts_:
                if len(group) == 1:
                    parts += group[0]
                else:
                    for possibility in group:
                        if len([p for p in possibility if p != Data()]) == len(possibility):
                            parts += possibility
                            break
            parts = [x for part in parts for x in part.plainOrdered()]

        if assertSize and len(parts) != spaceOrder.dim:
            raise Exception("Size mismatch while plaining out data {} into {}: {} != {}".format(self, spaceOrder,
                                                                                                len(parts), spaceOrder.dim))
        return parts

    def npPlain(self, spaceOrder=None, assertSize=True, fill=False):
        return np.array(self.plain())

    # Projections
    def singleSpaceComposante(self, singleSpace):
        flatten = self.flat()
        pm = ([d for d in flatten if d.space == singleSpace] + [Data()])[0]
        if not pm and singleSpace.rowAggregation:
            for possibility in singleSpace.rows:
                spaces = possibility.flatColsWithMultiRows
                projection = [self.singleSpaceComposante(s) for s in spaces]
                projection = [p for p in projection if p != Data()]
                if len(spaces) == len(projection):
                    return self.__class__._vectorClass_(singleSpace, [v for p in projection for v in p.valueOrdered])
        if not pm:
            pm = ([d for d in flatten if d.space.native == singleSpace.native and d.space.kind == singleSpace.kind]
                  + [Data()])[0]
        if not pm:
            pm = ([d for d in flatten if d.space.native ==
                   singleSpace.native] + [Data()])[0]

        return pm

    def projection(self, space, allowAbstraction=True):
        parts = [self.singleSpaceComposante(s)
                 for s in space.flatColsWithMultiRows]
        # if self.abstract:
        #     parts = [self.singleSpaceComposante(s) for s in space.flatColsWithMultiRows]
        # else:
        #     parts_ = [[list(map(self.singleSpaceComposante, possibility)) for possibility in group] for group in space.groupedCols]
        #     # parts = [part for group in parts for part in group if len([p for p in group if p != Data()]) == len(group)]
        #     parts = []
        #     for group in parts_:
        #         if len(group) == 1:
        #             parts += group[0]
        #         else:
        #             for possibility in group:
        #                 if len([p for p in possibility if p != Data()]) == len(possibility):
        #                     parts += possibility
        #                     break
        return self.__class__._vectorClass_(*[p for p in parts if p != Data()])

    # Entity
    def _findSpaceEntity(self, space, entity):
        if not space.rowAggregation:
            return self.singleSpaceComposante(space)
        parts = [[p for p in possibility if p.boundProperty.entity == entity]
                 for possibility in space.groupedCols[0]]
        part = [
            possibility for group in parts for possibility in group if possibility][0]
        return self.__class__._vectorClass_(part, np.array(self.valueOrdered)[self.space.columnsFor(space)])

    def applyTo(self, entity):
        # TODO: can only be applied to one entity abstraction, needs to be extended
        if not self.abstract:
            return self
        parts = [self._findSpaceEntity(s, entity)
                 for s in self.space.flatColsWithMultiRows]
        return self.__class__._vectorClass_(*parts)

    # Changing data
    def extends(self, other):
        if other is None:
            return self
        assert isinstance(other, Data)
        return self.__class__(*(self._parts + other._parts))

    def update(self, data, kindSensitive=True):
        if len(self) == 0:
            return self

        flatten = self.flat()
        flattenNew = data.convertTo(self._parts[0].space.spaceManager).flat()

        for newPart in flattenNew:
            for i, part in enumerate(flatten):
                if part.space.matches(newPart.space, kindSensitive=kindSensitive):
                    flatten[i] = newPart.__class__(
                        newPart.space, newPart.value)
                    break

        return self.__class__._vectorClass_(*[p for p in flatten])
    
    def convertTo(self, spaceManager=None, kind=None, toData=None):
        return self.__class__(*[part.convertTo(spaceManager=spaceManager, kind=kind, toData=toData)
                                for part in self._parts])


    def asTemplate(self, values):
        return self.__class__(self.space, values)

    @staticmethod
    def plainData(obj, spaceOrder=None, assertSize=True, fill=False):
        if isinstance(obj, SingleData):
            return list(obj.value)
        if isinstance(obj, Data):
            return obj.plainOrdered(spaceOrder, assertSize=assertSize, fill=fill)
        if isinstance(obj, list) and obj and (isinstance(obj[0], Data) or isinstance(obj[0], SingleData)):
            return [Data.plainData(o, spaceOrder=spaceOrder, assertSize=assertSize, fill=fill) for o in obj]
        return obj

    @staticmethod
    def npPlainData(obj, spaceOrder=None, assertSize=True, fill=False):
        val = Data.plainData(obj, spaceOrder=spaceOrder,
                             assertSize=assertSize, fill=fill)
        if isinstance(val, list):
            return np.array(val)
        return val

    def toStr(self, short=False):
        if short:
            return '\n    '.join([p.toStr() for p in self._parts])
        return "{}[{}]".format(self.__class__.__name__, self.toStr(short=True))

    def __repr__(self):
        return self.toStr()


class SingleData(Data):
    """
    Represents a space data
    :param space: data space
    :param value: data value
    """

    def __init__(self, space, value):
        self.space = space
        self.value = value if isinstance(value, list) else [value]
        self.valueOrdered = self.value
        self._parts = [self]
        if len(self.value) != self.space.dim:
            raise Exception("Size mismatch: space {} is {}d and the data value {} is {}d"
                            .format(self.space, self.space.dim, self.value, len(self.value)))
        self.relative = self.space.relative
        self.abstract = self.space.abstract

    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['space', 'value', 'valueOrdered', 'relative'], exportPathType=True)
        return dict_

    """
    Get the associated space (no CompositeSpace)
    :return SingleSpace instance
    """
    # def get_space(self):
    #     return self.space

    """
    Check whether this data is a vector or not
    :return False
    """

    def isVector(self):
        return False

    # Operators
    def __neg__(self):
        return self.__class__(self.space, [-v for v in self.value]).setRelative(self.relative)

    def __mul__(self, value):
        return self.__class__(self.space, [v*value for v in self.value]).setRelative(self.relative)

    def __add__(self, other):
        if other.space is None:
            raise ValueError('Spaces should be the same for both data')
        if not self.space.matches(other.space, kindSensitive=False):
            raise ValueError('Spaces (or native spaces) should be the same for both data (first is in {} and second is in {})'
                             .format(self.space.name, other.space.name))
        d = self.__class__(
            self.space, [v1 + v2 for v1, v2 in zip(self.value, other.value)])
        d.setRelative(self.relative and other.relative)
        return d

    def __sub__(self, other):
        if other.space is None:
            raise ValueError('Spaces should be the same for both data')
        if not self.space.matches(other.space, kindSensitive=False):
            raise ValueError('Spaces (or native spaces) should be the same for both data (first is in {} and second is in {})'
                             .format(self.space.name, other.space.name))
        return self.__class__(self.space, [v1 - v2 for v1, v2 in zip(self.value, other.value)]).setRelative(True)
    
    def setRelative(self, relative=None):
        if relative is not None:
            self.relative = relative
        return self
    
    def absoluteData(self, state):
        value = self.value
        if self.relative:
            for s in state:
                if s.space.matches(self.space):
                    value = [v1 + v2 for v1, v2 in zip(self.value, s.value)]
                    break
        return self.__class__(self.space, value).setRelative(False)

    def relativeData(self, state):
        value = self.value
        if not self.relative:
            for s in state:
                if s.space.matches(self.space):
                    value = [v1 - v2 for v1, v2 in zip(self.value, s.value)]
                    break
        return self.__class__(self.space, value).setRelative(True)
    
    def clone(self):
        return self.__class__(self.space, self.value).setRelative(self.relative)

    # Flatting
    def flat(self):
        return self

    def plain(self):
        return self.value

    def plainOrdered(self):
        return self.value

    # Changing data
    def convertTo(self, spaceManager=None, kind=None, toData=None):
        spaceManager = spaceManager if spaceManager else self.space.spaceManager
        return self.__class__(spaceManager.convertSpace(self.space, kind=kind, toData=toData),
                              self.value).setRelative(self.relative)

    def toStr(self, short=False):
        return "{}{}{}".format(self.space.toStr(True), '±' if self.relative else '',
                               ', '.join(["{:.3f}".format(v) for v in self.value]))

    def __repr__(self):
        return "{}({}{})".format(self.__class__.__name__, self.space.toStr(True),
                                 ["{:.3f}".format(v) for v in self.value])


class DataSequence(Serializable):
    """
    Represents a sequence data
    :param parts, ...: the sub data items
    """

    def __init__(self, *args):
        self._parts = list(args)

    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['_parts'], exportPathType=True)
        return dict_

    # Operators
    def append(self, part):
        self._parts.append(part)

    def __iter__(self):
        return self._parts.__iter__()

    def __getitem__(self, arg):
        return self._parts[arg]

    def __len__(self):
        return len(self._parts)
    
    def clone(self):
        return self.__class__(*[part.clone() for part in self._parts])

    def __repr__(self):
        return "{}{}".format(self.__class__.__name__, self._parts)

    def convertTo(self, spaceManager=None, kind=None, toData=None):
        return self.__class__(*[part.convertTo(spaceManager, kind, toData) for part in self._parts])


class Goal(Data):
    """
    Represents a Goal
    """
    pass


class SingleAction(SingleData):
    """
    Represents an Action in 1 SingleSpace
    """
    pass


# Action : set of simulatneous sub actions
class Action(Data):
    """
    Represents an Action in multiple Spaces
    """
    pass
    #def __repr__(self):
    #return "{}({})".format(self.__class__.__name__, '; '.join(["{}:{}".format(a.space.id, a.value) for a in self.flat()]))


# ActionList : set of successives actions
class ActionList(DataSequence):
    """
    Represents a sequence of ActionList in multiple Spaces
    """
    pass


class SingleObservation(SingleData):
    """
    Represents an SingleObservation in 1 SingleSpace
    """
    pass


class Observation(Data):
    """
    Represents an SingleObservation in multiple Spaces
    """
    pass


class Point(Data):
    """
    Represents an SingleObservation in multiple Spaces
    """
    pass


Data._vectorClass_ = Data
Observation._vectorClass_ = Observation
SingleObservation._vectorClass_ = Observation
Action._vectorClass_ = Action
SingleAction._vectorClass_ = Action

