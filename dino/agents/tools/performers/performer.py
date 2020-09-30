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

    MAX_DERIVE = 0.1

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
        r = self.__perform(path, config)
        return r

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
        # print('((START))')
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
        maxReplanning = 2
        maxDistance = 7.
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
                    newPath, _ = self.agent.planner.plan(
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

                # print('=== HEY')
                # print(node)
                # print(node.action.space.primitive())
                # print(node.execution)

                if not node.action.space.primitive() and not node.execution:
                    state = None
                    if node.parent:
                        state = node.parent.state
                    try:
                        node.execution, _ = self.agent.planner.plan(
                            node.action, state=state)
                    except ActionNotFound:
                        pass
                    if not node.execution:
                        self.logger.warning(
                            f'Iter {i}: failed to break down non primitive action')
                        break

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
                    primitiveActionExecuted = action
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
                    # print(f'#### {results[-1]}')

                # Check Distance
                # print('Miaou')
                if absoluteGoal:
                    distance = absoluteGoal.relativeData(
                        self.environment.state()).norm()
                    relative = absoluteGoal.relativeData(self.environment.state())
                    # print(absoluteGoal, distance)
                    self.logger.debug2(
                        f'Iter {i}: distance to goal {distance:.3f} (max {maxDistance:.3f}) ({relative})')
                    if distance < maxDistance:
                        self.logger.debug(
                            f'Iter {i}: close enough to goal!')
                        # print('((GOAL))')
                        return results
                
                # print('===')
                # print(node.absPos)
                if node.absPos:
                    derive = node.absPos.relativeData(
                        self.environment.state(), ignoreRelative=True).norm()
                    maxDerive = self.MAX_DERIVE * node.absPos.space.maxDistance
                    # print(node.action)
                    # print(derive, maxDerive)
                    if derive > maxDerive:
                        self.logger.info(
                            f'Iter {i}: max derive exceeded ({derive} > {maxDerive}) trying to reach {node.goal} by doing {node.action}')
                        if replanning < maxReplanning:
                            self.logger.info(
                                f'Replanning...')
                            nodes = None
                            break
                        else:
                            self.logger.warning(
                                f'No replanning left!')
                    # print(f'Derive is {derive} {node.absPos} {node.absPos.relativeData(self.environment.state(), ignoreRelative=True)}')
                    # print(self.environment.state().context())
                    # print(f'Max {self.MAX_DERIVE * node.absPos.space.maxDistance}')
                i += 1
            nodes = None
        # if o:
        #     y = o.difference(oStart)
        #results.append(InteractionEvent(self.getIteration(), actionListExecuted, y))

        # print('((OVER))')
        return results

    def iterative(self):
        return range(self.iterations)
