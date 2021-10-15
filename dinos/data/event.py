from exlab.interface.serializer import Serializable

from dinos.data.data import Data, Action, Observation


# Episode -> list of InteractionEvents
class InteractionEvent(Serializable):
    """
    Represents an Event, composed by different actions and outcomes
    :param actions: an ActionList object
    :param outcomes: an Observation object
    """

    def __init__(self, iteration, actions=Data(), primitiveActions=Data(), outcomes=Data(), context=Data()):
        self.iteration = iteration

        self.actions = actions
        self.primitiveActions = primitiveActions
        self.outcomes = outcomes
        self.context = context

        self.actionsRegister = []
        self.primitiveActionsRegister = []
        self.outcomesRegister = []
        self.contextRegister = []

        # Check if action is also present in the outcomes
        actions = self.actions.flat()
        onlyOutcomes = self.outcomes.flat()
        for outcome in list(onlyOutcomes):
            for action in actions:
                if outcome.space.matches(action.space):
                    action.value = outcome.value
                    onlyOutcomes.remove(outcome)

        # Check for no parameter actions (and set it to 1)
        # for action in actions:
        #     if len(action.value) == 0:
        #         action.value = [1]
        self.actions = Action(*actions)
        self.onlyOutcomes = Observation(*onlyOutcomes)
        self.allActions = None

    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['actions', 'primitiveActions', 'outcomes', 'context',
                                            'actionsRegister', 'primitiveActionsRegister', 'outcomesRegister',
                                            'contextRegister'])
        return dict_

    # @classmethod
    # def deserialize(cls, dict_, dataset, spaces, loadResults=True):
    #     a = [next(i for i in spaces if i.name == name or i.id == name) for name in dict_['actions']]
    #     y = [next(i for i in spaces if i.name == name or i.id == name) for name in dict_['outcomes']]
    #     c = [next(i for i in spaces if i.name == name or i.id == name) for name in dict_.get('context', [])]
    #     obj = cls(dataset, a, y, c)
    #     return obj

    def clone(self):
        return self.__class__(self.iteration,
                              self.actions.clone(),
                              self.primitiveActions.clone(),
                              self.outcomes.clone(),
                              self.context.clone())

    '''def flatten(self):
        return (self.action.flatten(), self.outcomes.flatten())

    @staticmethod
    def from_flat(data):
        return InteractionEvent(*data)'''

    def incrementIteration(self, n):
        self.iteration += n

    def convertTo(self, spaceManager=None, kind=None, toData=None):
        self.actions = self.actions.convertTo(
            spaceManager=spaceManager, kind=kind, toData=toData)
        self.primitiveActions = self.primitiveActions.convertTo(
            spaceManager=spaceManager, kind=kind, toData=toData)
        self.outcomes = self.outcomes.convertTo(
            spaceManager=spaceManager, kind=kind, toData=toData)
        self.onlyOutcomes = self.onlyOutcomes.convertTo(
            spaceManager=spaceManager, kind=kind, toData=toData)
        self.context = self.context.convertTo(
            spaceManager=spaceManager, kind=kind, toData=toData)
        
        allActions = self.actions.flat() + self.primitiveActions.flat() + self.onlyOutcomes.flat()
        self.allActions = Action(*allActions)

    def addToSpaces(self, cost=1.0):
        for point in self.actions.flat():
            idAction = point.space.addPoint(point, self.iteration, action=True)
            self.actionsRegister.append((point.space.id, idAction))

        if not self.actions == self.primitiveActions:
            for point in self.primitiveActions.flat():
                idAction = point.space.addPoint(point, self.iteration, action=True)
                self.primitiveActionsRegister.append((point.space.id, idAction))

        for point in self.onlyOutcomes.flat():
            idOutcome = point.space.addPoint(point, self.iteration, cost)
            self.outcomesRegister.append((point.space.id, idOutcome))

        if self.context:
            for point in self.context.flat():
                idContext = point.space.addPoint(point, self.iteration, cost)
                self.contextRegister.append((point.space.id, idContext))

    @staticmethod
    def incrementList(list_, currentIteration):
        if not list_:
            return 0
        n = max([event.iteration for event in list_]) + 1
        for event in list_:
            event.incrementIteration(currentIteration)
        return n

    def __repr__(self):
        return f'{self.__class__.__name__}#{self.iteration}({self.actions}, {self.outcomes}, {self.context})'
