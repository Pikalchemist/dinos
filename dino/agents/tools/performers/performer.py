'''
    File name: performer.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

from exlab.modular.module import Module

from dino.utils.move import MoveConfig
from dino.data.event import InteractionEvent
from dino.data.data import ActionList, Data
from dino.data.space import SpaceKind, FormatParameters
from dino.data.path import ActionNotFound


class Performer(Module):
    """
    Executes Paths created by a planner or ActionList
    """

    def __init__(self, agent, options={}):
        super().__init__('Performer', agent)
        self.logger.tag = 'performer'

        self.agent = agent
        self.environment = self.agent.environment
        self.options = options

        self.iterations = max(1, options.get('iterations', 1))

    def perform(self, paths, config=MoveConfig()):
        """
        Tests a specific Paths and stores consequences in memory.
        :param paths: Paths object (created by a planner)
        """
        groupedActionList = paths.getGroupedActionList()
        # print(groupedActionList)
        return self.__perform(groupedActionList, config)

    def performGoal(self, goal, paths=None, config=MoveConfig()):
        if paths:
            groupedActionList = paths.getGroupedActionList()
        else:
            groupedActionList = None
        return self.__perform(groupedActionList, config, goal=goal)

    def performActions(self, actionList, config=MoveConfig()):
        """
        Tests a specific action list and stores consequences in memory.
        :param actionList: ActionList
        """
        return self.__perform([(None, actionList)], config)

    def __perform(self, groupedActionList, config, goal=None):
        """
        Tests a specific action list and stores consequences in memory.
        """
        results = []
        actionListExecuted = ActionList()
        # mem = []

        if not config.allowReplanning:
            goal = None
        if goal:
            absoluteGoal = goal.absoluteData(self.environment.state())
            # print("======== GOAL ========", absoluteGoal)

        replanning = 0
        maxReplanning = 10
        maxDistance = 3.

        formatParameters = FormatParameters()
        oStart = self.agent.observe(formatParameters=formatParameters)
        oPrevious = self.agent.observe(formatParameters=formatParameters)
        o = None
        running = True
        # goal = None

        i = 0
        while running:
            if goal:
                if not groupedActionList:
                    self.logger.debug2(
                        f'Iter {i}: no more action to execute... trying to replan new ones')
                    replanning += 1
                    if replanning > maxReplanning:
                        self.logger.warning(
                            f'Iter {i}: out of replanning')
                        break
                    try:
                        paths = self.agent.planner.plan(
                            absoluteGoal, model=config.model)
                    except ActionNotFound:
                        break
                    # distance = paths.length()
                    # if distance < 2.:
                    #     break
                    # print('Replanning...')
                    groupedActionList = paths.getGroupedActionList()
                # groupedActionList = [groupedActionList[0]]
            # print('>>>>>>>>>>', groupedActionList)
            for node, actionList in groupedActionList:
                if not running:
                    break
                # print('WEEEESH', node, actionList)
                # print(len(actionList))
                for action in actionList:
                    # print("=========")
                    # print(node.goal)
                    # print(action)
                    # print(node)
                    # print("===")
                    # print(actionList)
                    actionExecuted = node.goal if node and node.goal else action
                    primitiveActionExecuted = action if node and node.goal else Data()
                    actionListExecuted.append(actionExecuted)

                    # print(actions[i])
                    self.agent._performAction(action, config=config)
                    # print(action)
                    # print("Tasks : " + str(spaces))
                    # print(y_list)
                    #real_actions += actions[i].tolist()
                    # print("---")
                    o = self.agent.observe(formatParameters=formatParameters)
                    y = o.difference(oPrevious)
                    results.append(InteractionEvent(self.environment.counter.last,
                                                    actionExecuted,
                                                    primitiveActionExecuted,
                                                    y,
                                                    oPrevious.convertTo(kind=SpaceKind.PRE)))
                    oPrevious = o

                    # Check Distance
                    if goal:
                        distance = absoluteGoal.relativeData(
                            self.environment.state()).norm()
                        relative = absoluteGoal.relativeData(self.environment.state())
                        self.logger.debug2(
                            f'Iter {i}: distance to goal {distance:.3f} (max {maxDistance:.3f}) ({relative})')
                        if distance < maxDistance:
                            self.logger.debug(
                                f'Iter {i}: close enough to goal!')
                            running = False
                            break
                i += 1
            if goal:
                groupedActionList = None
            else:
                break
        if o:
            y = o.difference(oStart)
        #results.append(InteractionEvent(self.getIteration(), actionListExecuted, y))

        return results

    def iterative(self):
        return range(self.iterations)
