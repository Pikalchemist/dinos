import copy

from dino.data.data import Data


class State(object):
    def __init__(self, environment, values=[], dataset=None):
        self.environment = environment
        if not isinstance(values, list):
            values = [values]
        self.values = [part for value in values for part in value.flat()]
        if dataset:
            self.values = [part.convertTo(dataset) for part in self.values]
        self.update()

    def __repr__(self):
        return f"State ({self.values})"

    def __iter__(self):
        return self.values.__iter__()

    def update(self):
        self._context = Data(*self.values)

    def apply(self, action, dataset, overwrite=None):
        variations = {}
        actionSpace = action.space
        context = self.context()
        for model in dataset.models:
            if model.isCoveredByActionSpaces(actionSpace):
                result = model.forward(action, context)
                parts = result[0].flat()
                for part in parts:
                    variations[part.space] = part

        for i, part in enumerate(self.values):
            if part.space in variations.keys():
                self.values[i] = part + variations[part.space]
            if overwrite:
                overpart = overwrite.projection(part.space)
                if overpart:
                    self.values[i] = overpart

        self.update()
        return self

    def context(self):
        return self._context

    def difference(self, previous, space=None):
        diff = self._context.difference(previous._context)
        if space:
            diff = diff.projection(diff)
        return diff

    def copy(self):
        return self.__class__(self.environment, list(self.values))
