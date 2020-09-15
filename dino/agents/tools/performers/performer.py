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


class Performer(Module):
    """
    Executes Paths created by a planner or ActionList
    """

    def __init__(self, agent, options={}):
        super().__init__('Performer')
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
        iteration = 0
        o = None
        running = True
        # goal = None

        while running:
            if goal:
                if not groupedActionList:
                    replanning += 1
                    if replanning > maxReplanning:
                        print('Out of replanning')
                        break
                    paths = self.agent.planner.plan(
                        absoluteGoal, model=config.model)
                    # distance = paths.length()
                    # if distance < 2.:
                    #     break
                    print('Replanning...')
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
                    results.append(InteractionEvent(iteration,
                                                    actionExecuted,
                                                    primitiveActionExecuted,
                                                    y,
                                                    oPrevious.convertTo(kind=SpaceKind.PRE)))
                    iteration += 1
                    oPrevious = o

                    # Check Distance
                    if goal:
                        distance = absoluteGoal.relativeData(
                            self.environment.state()).norm()
                        print('DISTANCE', distance, absoluteGoal.relativeData(
                            self.environment.state()))
                        if distance < maxDistance:
                            running = False
                            break
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
