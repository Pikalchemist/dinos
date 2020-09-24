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
from dino.data.path import Path, ActionNotFound

from dino.agents.tools.planners.planner import PlanSettings


class Performer(Module):
    """
    Executes Paths created by a planner
    """

    def __init__(self, agent, options={}):
        super().__init__('Performer', agent)
        self.logger.tag = 'performer'

        self.agent = agent
        self.environment = self.agent.environment
        self.options = options

        self.iterations = max(1, options.get('iterations', 1))

    def perform(self, path, config=MoveConfig()):
        """
        Tests a specific Path and stores consequences in memory.
        :param path: Path object (created by a planner)
        """
        return self.__perform(path, config)

    # def performGoal(self, goal, path=None, config=MoveConfig()):
    #     if paths:
    #         groupedActionList = paths.getGroupedActionList()
    #     else:
    #         groupedActionList = None
    #     return self.__perform(path, config)

    def performActions(self, actionList, goal=None, config=MoveConfig()):
        """
        Tests a specific action list and stores consequences in memory.
        :param actionList: ActionList
        """
        path = self.agent.planner.planActions(actionList)
        return self.__perform(path, config)

    def __perform(self, path, config, i=0):
        """
        Tests a specific action list and stores consequences in memory.
        """
        results = []
        # mem = []

        if path.goal:
            absoluteGoal = path.goal.absoluteData(self.environment.state())
        else:
            absoluteGoal = None
        if not config.allowReplanning:
            absoluteGoal = None
            # print("======== GOAL ========", absoluteGoal)

        # goal = None

        replanning = 0
        maxReplanning = 10
        maxDistance = 3.
        nodes = list(path.nodes)

        actionListExecuted = ActionList()
        formatParameters = FormatParameters()
        plannerSettings = PlanSettings()
        observationsPrevious = None
        # oStart = self.agent.observe(formatParameters=formatParameters)
        while True:
            if absoluteGoal and not nodes:
                self.logger.debug2(
                    f'Iter {i}: no more action to execute... trying to replan new ones')
                replanning += 1
                if replanning > maxReplanning:
                    self.logger.warning(
                        f'Iter {i}: out of replanning')
                    break
                try:
                    newPath = self.agent.planner.plan(
                        absoluteGoal, settings=plannerSettings)
                    nodes = newPath.nodes
                except ActionNotFound:
                    break
            
            if not nodes:
                break

            for node in nodes:
                if node.context:
                    results += self.__perform(node.context, config)
                    observationsPrevious = None
                if node.execution:
                    results += self.__perform(node.execution, config)
                    observationsPrevious = None
                if not node.context and not node.execution:
                    action = node.action
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
                    if observationsPrevious is None:
                        observationsPrevious = self.agent.observe(formatParameters=formatParameters)
                    self.agent._performAction(action, config=config)
                    # print(action)
                    # print("Tasks : " + str(spaces))
                    # print(y_list)
                    #real_actions += actions[i].tolist()
                    # print("---")
                    observations = self.agent.observe(formatParameters=formatParameters)
                    y = observations.difference(observationsPrevious)
                    results.append(InteractionEvent(self.environment.counter.last,
                                                    actionExecuted,
                                                    primitiveActionExecuted,
                                                    y,
                                                    observationsPrevious.convertTo(kind=SpaceKind.PRE)))
                    observationsPrevious = observations

                    # Check Distance
                    if absoluteGoal:
                        distance = absoluteGoal.relativeData(
                            self.environment.state()).norm()
                        relative = absoluteGoal.relativeData(self.environment.state())
                        self.logger.debug2(
                            f'Iter {i}: distance to goal {distance:.3f} (max {maxDistance:.3f}) ({relative})')
                        if distance < maxDistance:
                            self.logger.debug(
                                f'Iter {i}: close enough to goal!')
                            return results
                i += 1
            nodes = None
        # if o:
        #     y = o.difference(oStart)
        #results.append(InteractionEvent(self.getIteration(), actionListExecuted, y))

        return results

    def iterative(self):
        return range(self.iterations)
