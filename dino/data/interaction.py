from exlab.interface.serializer import Serializable

from dino.data.data import Action, Observation


# Episode -> list of InteractionEvents
class InteractionEvent(Serializable):
    """
    Represents an Event, composed by different actions and outcomes
    :param actions: an ActionList object
    :param outcomes: an Observation object
    """

    def __init__(self, iteration, actions=None, primitiveActions=None, outcomes=None, context=None):
        self.iteration = iteration

        self.actions = actions
        self.primitiveActions = primitiveActions
        self.outcomes = outcomes
        self.context = context

        self.actionsRegister = []
        self.primitiveActionsRegister = []
        self.outcomesRegister = []
        self.contextRegister = []

        self.id = -1

        # Check if action is also present in the outcomes
        actions = self.actions.flat()
        outcomes = self.outcomes.flat()
        for outcome in list(outcomes):
            for action in actions:
                if outcome.space.matches(action.space):
                    action.value = outcome.value
                    outcomes.remove(outcome)

        # Check for no parameter actions (and set it to 1)
        # for action in actions:
        #     if len(action.value) == 0:
        #         action.value = [1]
        self.actions = Action(*actions)
        self.outcomes = Observation(*outcomes)

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
        self.context = self.context.convertTo(
            spaceManager=spaceManager, kind=kind, toData=toData)

    @staticmethod
    def incrementList(list_, currentIteration):
        if not list_:
            return 0
        n = max([event.iteration for event in list_]) + 1
        for event in list_:
            event.incrementIteration(currentIteration)
        return n

    def __repr__(self):
        return '{}#{}({}, {}, {})'.format(self.__class__.__name__, self.iteration, self.actions, self.outcomes,
                                          self.context)
