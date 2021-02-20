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
from dino.agents.tools.policies.policy import LearningPolicy


class Performer(Module):
    """
    Executes Paths created by a planner
    """

    MAX_DERIVE = 0.04
    MAX_DISTANCE = 0.01

    def __init__(self, agent, options={}):
        super().__init__('Performer', agent, loggerTag='performer')

        self.agent = agent
        self.environment = self.agent.environment
        self.options = options

        self.iterations = max(1, options.get('iterations', 1))

    def perform(self, path, config=MoveConfig()):
        """
        Tests a specific Path and stores consequences in memory.
        :param path: Path object (created by a planner)
        """
        r = self.__perform(path, config)[0]
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
        return self.__perform(path, config)[0]

    def __perform(self, path, config, i=0, depth=0):
        """
        Tests a specific action list and stores consequences in memory.
        """
        results = []
        lastResults = []
        # print('((START))')

        # for node in list(path.nodes):
        #     print(node)

        absoluteGoal = None
        if path.goal:
            absoluteGoal = path.goal.absoluteData(self.environment.state())
        if path.planSettings:
            path.planSettings.performing = True
        model = path.model
        # print("======== GOAL ========", absoluteGoal)

        # settings
        replanning = 0
        maxReplanning = 5
        maxDistance = 7.
        maxDerive = None
        if model:
            maxDistance = model.getPrecision(self.MAX_DISTANCE, 1.5)
            maxDerive = model.getPrecision(self.MAX_DERIVE, 2.5)
        nodes = list(path.nodes)

        formatParameters = FormatParameters()
        observationsPrevious = None
        distanceToGoal = None
        distanceToGoalSinglePass = None
        currentPos = None

        success = False
        fatal = False
        topReplanning = False
        while not success and not fatal:
            if not nodes and absoluteGoal and config.allowReplanning:
                if distanceToGoal:
                    if replanning == 0:
                        distanceToGoalSinglePass = distanceToGoal
                    # self.postPerforming(model, False, distanceToGoal)
                replanning += 1
                config.result.performerReplanning += 1
                if replanning > maxReplanning:
                    self.logger.warning(f'Iter (d{depth}) {i}: out of replanning')
                    break
                try:
                    self.logger.warning(f'Iter (d{depth}) {i}: no more action to execute... trying to replan new ones')
                    newPath, _ = self.agent.planner.plan(absoluteGoal, settings=path.planSettings)
                    if newPath:
                        nodes = newPath.nodes
                except ActionNotFound:
                    self.logger.warning(f'Iter (d{depth}) {i}: failed to plan new one')

            if not nodes:
                break

            for node in nodes:
                if node.context:
                    subResults, (fatal, topReplanning) = self.__perform(node.context, config, depth=depth+1)
                    if fatal:
                        break
                    results += subResults
                    observationsPrevious = None

                # print('=== HEY')
                # print(node)
                # print(node.action.space.primitive())
                # print(node.execution)

                if not node.action.space.primitive() and not node.execution:
                    try:
                        node.execution, _ = self.agent.planner.plan(node.action, settings=PlanSettings(performing=True))
                    except ActionNotFound:
                        pass
                    if not node.execution:
                        self.logger.warning(
                            f'Iter (d{depth}) {i}: failed to break down non primitive action {node.action}')
                        fatal = True
                        break

                if node.absPos:
                    currentPos, derive, currentState = self.checkStatus(node, node.parent.absPos)
                    self.logger.debug(
                        f'Iter (d{depth}) {i}: pre execution check: should be at {node.parent.absPos} and currently at {currentPos}, diff {derive:.4f}\nCurrent Pre State:\n{currentState.context()}')

                if node.execution:
                    observationsPrevious = self.agent.observe(formatParameters=formatParameters)
                    subResults, (fatal, topReplanning) = self.__perform(node.execution, config, depth=depth+1)
                    if fatal:
                        break
                    results += subResults
                    observations = self.agent.observe(formatParameters=formatParameters)
                    differences = observations.difference(observationsPrevious)
                    observationsPrevious = None
                else:
                    # self.logger.debug(f'Iter (d{depth}) {i}:\n{node.model.npForward(node.action, currentState.context())}')
                    event, differences, observationsPrevious = self.performAction(
                        node, observationsPrevious, formatParameters, config)
                    results.append(event)
                    lastResults.append(event)
                
                # print('=== ===')
                # print(node.state)
                # print(self.environment.state().context())

                # self.logger.debug(
                #     f'Iter (d{depth}) {i}:\n{node.ty0} Estimated state:\n{node.state.context()}\nCurrent New State:\n{self.environment.state().context()}')
                
                # self.logger.info(f'Iter (d{depth}) {i}: performing {node.action}')

                currentState = self.environment.state()
                # self.logger.warning(currentState)
                rdiff, rderive = None, -1
                if node.absPos:
                    currentPos, derive, currentState = self.checkStatus(node, node.absPos, currentState)

                    rdiff = differences.projection(node.absPos.space)  # if differences is not None else None
                    rderive = (rdiff - node.goal).norm()  # if differences is not None else -1.
                    if model:
                        model.updatePrecisionPerGoal(node.goal, rderive)
                    self.logger.debug(
                        f'Iter (d{depth}) {i}: wanting {node.absPos} and got {currentPos} \n     Diff {derive:.4f} doing {node.action} to get {node.goal} and got {rdiff} Diff {rderive:.4f}')

                # Check Distance
                # print('======Miaou======', absoluteGoal)
                if absoluteGoal:
                    relative = absoluteGoal.relativeData(currentState)
                    distanceToGoal = relative.norm()
                    if distanceToGoal < maxDistance and node == nodes[-1]:
                        self.logger.debug(
                            f'Iter (d{depth}) {i}: close enough to goal! {distanceToGoal:.3f} (max {maxDistance:.3f}) ({relative})')
                        success = True
                        break
                    else:
                        self.logger.debug(
                            f'Iter (d{depth}) {i}: distance to goal {distanceToGoal:.3f} (max {maxDistance:.3f}) ({relative})')
                
                # print('===')
                # print(node.absPos)
                if node.absPos and maxDerive:
                    currentPos, derive, currentState = self.checkStatus(node, node.absPos, currentState)
                    # print(node.action)
                    # print(derive, maxDerive)
                    if derive > maxDerive:
                        self.logger.info(
                            f'Iter (d{depth}) {i}: max derive exceeded ({derive:.4f} > {maxDerive:.4f}) trying to reach {node.goal} by doing {node.action} to get {node.goal} and got {rdiff} Diff {rderive:.4f}')
                        self.postPerforming(model, False, derive)
                        self.postPerforming(model, False, derive, True)
                        if depth == 0:
                            config.result.performerDerive.append((derive, maxDerive))
                        if replanning <= maxReplanning:
                            self.logger.info(f'Replanning...')
                            nodes = None
                            break
                        else:
                            self.logger.warning(f'No replanning left!')
                    # print(f'Derive is {derive} {node.absPos} {node.absPos.relativeData(self.environment.state(), ignoreRelative=True)}')
                    # print(self.environment.state().context())
                    # print(f'Max {self.MAX_DERIVE * node.absPos.space.maxDistance}')
                i += 1

                if self.agent.learningPolicy == LearningPolicy.EACH_ITERATION:
                    self.agent.addMemory(lastResults, config)
                    lastResults = []

            nodes = None
        # if o:
        #     y = o.difference(oStart)
        #results.append(InteractionEvent(self.getIteration(), actionListExecuted, y))

        # print('((OVER))')

        if distanceToGoal:
            if replanning == 0:
                distanceToGoalSinglePass = distanceToGoal
            if distanceToGoalSinglePass:
                self.postPerforming(model, success, distanceToGoalSinglePass, True)
            self.postPerforming(model, success, distanceToGoal * (1 + replanning * 0.1))
            if depth == 0:
                config.result.performerDistance = (distanceToGoal, maxDistance)

        return results, (fatal, topReplanning)
    
    def performAction(self, node, observationsPrevious, formatParameters, config):
        action = node.action
        actionExecuted = (node.goal if node and node.goal else action).clone()
        primitiveActionExecuted = action
        # actionListExecuted.append(actionExecuted)

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
        differences = observations.difference(observationsPrevious)
        event = InteractionEvent(self.environment.counter.last,
                                 actionExecuted,
                                 primitiveActionExecuted,
                                 differences,
                                 observationsPrevious.convertTo(kind=SpaceKind.PRE))
        observationsPrevious = observations

        return event, differences, observationsPrevious
    
    def checkStatus(self, node, goal, currentState=None):
        if currentState is None:
            currentState = self.environment.state()
        derive = goal.relativeData(currentState, ignoreRelative=True)
        currentPos = goal - derive
        distanceDerive = derive.norm()
        return currentPos, distanceDerive, currentState
    
    def postPerforming(self, model, success, distanceToGoal, singlePrecision=False):
        if model:
            model.updatePrecision(success, distanceToGoal, index=int(singlePrecision))

    def iterative(self):
        return range(self.iterations)
