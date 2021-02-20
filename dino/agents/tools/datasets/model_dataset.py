'''
    File name: dataset.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import numpy as np

from dino.data.space import SpaceKind
from dino.agents.tools.models.model import Model

from .dataset import Dataset


class ModelDataset(Dataset):
    """The dataset used to record actions, outcomes and procedures."""

    MODEL_PERF_COMPETENCE = 0.5
    MODEL_PERF_DURATION = 120

    def __init__(self, modelClass=Model, options={}):
        """
        options dict: parameters for the dataset
        """
        super().__init__(options=options)

        self.modelClass = modelClass
        self.models = []

        # self.iterationIds = []
        # self.iteration = -1

        # Contains the following keys
        # - min_y_best_locality: minimum of outcomes to keep in best locality searches
        # - min_a_best_locality: minimum of actions to keep in best locality searches
        self.options = options
        self.setOptions()

        # self.spacesHistory = DataEventHistory()
        # self.modelsHistory = DataEventHistory()

    # def gid(self):
    #     if self.moduleParent:
    #         return Serializer.make_gid(self, self.moduleParent.gid())
    #     return Serializer.make_gid(self)

    def _serialize(self, serializer):
        dict_ = super()._serialize(serializer)
        dict_.update(serializer.serialize(
            self, ['models', 'modelClass']))
        return dict_
    
    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls(serializer.deserialize(dict_.get('modelClass')), dict_.get('options', {}))
        return super()._deserialize(dict_, serializer, obj)

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        serializer.deserialize(dict_.get('models', []))
    
    # @classmethod
    # def _deserialize(cls, dict_, serializer, obj=None):
    #     if obj is None:
    #         obj = cls(dict_.get('storesData'),
    #                   options=dict_.get('options', {}))
    #     return obj

    # def _postDeserialize(self, dict_, serializer):
    #     serializer.set('spaceManager', self)
    #     for spaceData in dict_.get('spaces', []):
    #         serializer.deserialize(spaceData)

    # @classmethod
    # def _deserialize(cls, dict_, modelClass=Model, options=None, obj=None):
    #     if 'model' in options:
    #         from ..utils.loaders import DataManager
    #         modelClass = DataManager.loadType(
    #             options['model']['path'], options['model']['type'])
    #     obj = obj if obj else cls(
    #         modelClass=modelClass, options=dict_.get('options', {}))
    #     obj = Module._deserialize(dict_, options=options, obj=obj)
    #     obj = SpaceManager._deserialize(dict_, options=options, obj=obj)
    #     # Operations
    #     models = [obj.modelClass._deserialize(
    #         model, obj, spaces, loadResults=loadResults) for model in dict_.get('models', [])]
    #     return obj

    '''@classmethod
    def clone_without_content(cls, dataset):
        """Function that will copy an other dataset except without any data."""
        result = Dataset([], [], copy.deepcopy(dataset.options))
        for a_s in dataset.actionSpaces:
            result.actionSpaces.append([])
            for a in a_s:
                result.actionSpaces[-1].append(ActionSpace.clone_without_content(a))
        for y in dataset.outcomeSpaces:
            result.outcomeSpaces.append(OutcomeSpace.clone_without_content(y))
        return result'''
    
    # Triggered when new data are added
    def updated(self):
        self.invalidateCompetences()

    # Models
    def model(self, index):
        return next(s for s in self.models if s.id == index)
    
    def enabledModels(self):
        return [model for model in self.models if model.enabled]

    def registerModel(self, model):
        if model not in self.models:
            self.logger.info(f'New model added: {model}', tag='model')
            self.models.append(model)
            self.computeSpaces()
            model.createdSince = self.learner.iteration

    def unregisterModel(self, model):
        if model in self.models:
            self.logger.info(f'Model removed: {model}', tag='model')
            self.models.remove(model)
            self.computeSpaces()
            model.createdSince = -1

    def replaceModel(self, model, newModel):
        # del self.models[self.models.index(newModel)]
        # self.models[self.models.index(model)] = newModel
        self.unregisterModel(model)
        self.registerModel(newModel)
        newModel.continueFrom(model)

    def findModelByOutcomeSpace(self, outcomeSpace, models=None):
        return (self.findModelsByOutcomeSpace(outcomeSpace, models) + [None])[0]

    def findModelsByOutcomeSpace(self, outcomeSpace, models=None):
        models = models if models else self.enabledModels()
        return [m for m in models if m.coversOutcomeSpaces(outcomeSpace)]

    def findModelsByActionSpace(self, actionSpace, models=None):
        models = models if models else self.enabledModels()
        return [m for m in models if m.coversActionSpaces(actionSpace)]
    
    def competences(self, precise=False):
        return {model: model.competence(precise=precise) for model in self.models if model.enabled}
    
    def invalidateCompetences(self):
        for model in self.models:
            model.invalidateCompetences()

    # Graph
    def dependencyGraph(self, models=None):
        models = models if models else self.enabledModels()

        graph = {}
        for model in models:
            edges = set()
            for space in model.outcomeSpace.flatSpaces:
                edges.update(set(self.findModelsByActionSpace(space, models)))
            graph[model] = tuple(edges)

        return graph

    def isGraphCyclic(self, graph):
        path = set()
        visited = set()

        def visit(node):
            if node in visited:
                return False
            visited.add(node)
            path.add(node)
            for neighbour in graph.get(node, ()):
                if neighbour in path or visit(neighbour):
                    return True
            path.remove(node)
            return False

        return any(visit(node) for node in graph)

    # Spaces
    def controllableSpaces(self, spaces=None, merge=False, performant=False, explorable=False, convert=True):
        return self.__controllableSpaces(True, spaces, merge, performant, explorable, convert=convert)

    def nonControllableSpaces(self, spaces=None, merge=False, performant=False, explorable=False, convert=True):
        return self.__controllableSpaces(False, spaces, merge, performant, explorable, convert=convert)

    def __controllableSpaces(self, controllable, spaces=None, merge=False, performant=False, explorable=False, convert=True):
        spaces = spaces if spaces else self.actionExplorationSpaces
        if convert:
            if type(spaces) in (list, set):
                spaces = [space.convertTo(kind=SpaceKind.BASIC) for space in spaces]
            else:
                spaces = spaces.convertTo(kind=SpaceKind.BASIC)
        spaces = [space for space in spaces if self.controllableSpace(space, performant, explorable) == controllable]
        if merge:
            return self.multiColSpace(spaces)
        return spaces

    def controllableSpace(self, space, performant=False, explorable=False):
        if space is None or space.null():
            return False
        if space.primitive():
            return True
        models = self.findModelsByOutcomeSpace(space.convertTo(kind=SpaceKind.BASIC))
        if performant:
            models = [model for model in models if model.performant(self.MODEL_PERF_COMPETENCE, self.MODEL_PERF_DURATION)]
        if explorable:
            models = [model for model in models if model.explorable()]
        return len(models) > 0

    def controlContext(self, goalContext, currentContext):
        if not currentContext or not goalContext:
            return goalContext

        space = goalContext.space
        currentContext = currentContext.projection(space, kindSensitive=True)

        nonControllable = currentContext.projection(
            self.nonControllableSpaces(space, merge=True, convert=False), kindSensitive=True)
        controllable = goalContext.projection(
            self.controllableSpaces(space, merge=True, convert=False), kindSensitive=True)
        context = nonControllable.extends(controllable) 

        return context

    def checkSpaceChanges(self, spaces, action, context=None, ratio=False, precision=0.02):
        # space.maxDistance * precision
        return np.sum([self.computeSpaceChange(space, action, context, ratio=ratio) for space in spaces])

    def computeSpaceChange(self, space, action, context=None, ratio=False):
        actionSpace = action.space
        models = [m for m in self.enabledModels() if m.coversActionSpaces(actionSpace) and m.coversOutcomeSpaces(space)]
        outcomes = [m.forward(action, context)[0].projection(space) for m in models]
        distance = np.sum([y.norm() for y in outcomes])
        if ratio:
            return distance / space.maxDistance
        return distance

    # Data
    def setData(self, spaces, models, addHistory=True):
        if len(self.spaces) + len(self.models) > 0:
            return
        self.models = models
        self.spaces = spaces
        self.computeSpaces()

        # Add default spaces and models to history
        # if addHistory:
        #     self.spacesHistory.append(self.getIteration(), DataEventKind.ADD,
        #                               [("{}".format(s.id), s.serialize()) for s in self.spaces])
        #     self.modelsHistory.append(self.getIteration(), DataEventKind.ADD,
        #                               [("{}".format(m.id), m.serialize()) for m in self.models])

    def setOptions(self):
        if 'min_y_best_locality' not in self.options.keys():
            self.options['min_y_best_locality'] = 5
        if 'min_a_best_locality' not in self.options.keys():
            self.options['min_a_best_locality'] = 4
