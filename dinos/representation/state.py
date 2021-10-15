import copy

from exlab.utils.io import parameter
from dinos.data.data import Data


class State(object):
    def __init__(self, environment, values=[], dataset=None, featureMap=None, enforceDataset=True):
        self.environment = environment
        self.dataset = dataset
        self.featureMap = featureMap
        if not isinstance(values, list):
            values = [values]
        self.values = [part for value in values for part in value.flat()]
        if dataset and enforceDataset:
            self.values = [part.convertTo(dataset) for part in self.values]
        self.update()

    def __repr__(self):
        return f"State ({self.values})"

    def __iter__(self):
        return self.values.__iter__()

    def update(self):
        if self.featureMap:
            self.featureMap.update(self.values, self.dataset)
        self._context = Data(*self.values)
    
    def consequences(self, controllable, excludeSpaces=None):
        excludeSpaces = list(parameter(excludeSpaces, set()))

        consequences = []
        controllables = [controllable]
        while controllables:
            newControllables = []
            for controllable in controllables:
                newControllables += self._controllableConsequences(controllable, excludeSpaces)
            controllables = newControllables
            consequences +=  controllables
        
        return consequences
        

    def _controllableConsequences(self, controllable, excludeSpaces):
        consequences = []
        controllableSpace = controllable.space
        context = self.context()

        for model in self.dataset.enabledModels():
            if model.isCoveredByActionSpaces(controllableSpace):
                result, _ = model.forward(controllable, context)
                if result:
                    for part in result.flat():
                        if part.space not in excludeSpaces:
                            consequences.append(part)
                            excludeSpaces.append(part)

        return consequences

    def apply(self, controllable, excludeSpaces=set(), overwrite=None):
        consequences = self.consequences(controllable, excludeSpaces)

        variations = {c.space: c for c in consequences}

        for i, part in enumerate(self.values):
            if part.space in variations.keys():
                self.values[i] = part + variations[part.space]
            if overwrite:
                overpart = overwrite.projection(part.space)
                if overpart:
                    self.values[i] = overpart



        # variations = {}
        # actionSpace = controllable.space
        # context = self.context()
        # for model in dataset.enabledModels():
        #     if model.isCoveredByActionSpaces(actionSpace):
        #         result, _ = model.forward(controllable, context)
        #         if result:
        #             parts = result.flat()
        #             for part in parts:
        #                 variations[part.space] = part
        #         # else:
        #         #     # print(f'======= {model} {action} {context}')
        #         #     result[0].flat()

        # for i, part in enumerate(self.values):
        #     if part.space in variations.keys():
        #         self.values[i] = part + variations[part.space]
        #     if overwrite:
        #         overpart = overwrite.projection(part.space)
        #         if overpart:
        #             self.values[i] = overpart

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
        return self.__class__(self.environment, list(self.values), self.dataset, self.featureMap, enforceDataset=False)
