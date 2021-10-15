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

    def __init__(self, agent, options={}):
        super().__init__('Performer', agent, loggerTag='performer')

        self.agent = agent
        self.environment = self.agent.environment
        self.options = options

        self.iterations = max(1, options.get('iterations', 1))
        self.lastSequence = None

    def perform(self, path, config=MoveConfig()):
        """
        Tests a specific Path and stores consequences in memory.
        :param path: Path object (created by a planner)
        """
        r = self.__perform(path, config)[0]
        return r

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
        result, p = Performance.perform(self, path, config, i=i, depth=depth)
        self.lastSequence = p.lastSequence
        return result

    def iterative(self):
        return range(self.iterations)


class PerformedInfo(object):
    def __init__(self, message):
        self.message = message
    
    def __repr__(self):
        return self.message


class PerformedSequence(object):
    def __init__(self, node, depth):
        self.node = node
        self.depth = depth

        self.startingPosition = None
        self.startingDerive = None
        self.endingPosition = None
        self.endingDerive = None

        self.predictedOutcome = None
        self.startingPredictionDerive = None
        self.endingPredictionDerive = None
    
    def __repr__(self):
        spaces = '  ' * self.depth
        return f'{spaces} Prediction({self.predictedOutcome} estimated ±{self.startingPredictionDerive:.4f} and got ±{self.endingPredictionDerive:.4f})  \
Reality({self.startingPosition.plain()} ±{self.startingDerive:.4f})->({self.endingPosition.plain()} ±{self.endingDerive:.4f})'
    
    @property
    def action(self):
        return self.node.action
    
    @property
    def startingWantedPosition(self):
        return self.node.parent.absPos
    
    @property
    def endingWantedPosition(self):
        return self.node.absPos


class Performance(object):
    MAX_DERIVE = 0.04
    MAX_DISTANCE = 0.01
    RECOMPUTE_EXECUTION = False
    MAX_PREDICTED_LINEAR_ERROR = 0.002

    def __init__(self, performer, path, config, parent=None, i=0, depth=0):
        self.performer = performer
        self.path = path
        self.config = config
        self.depth = depth
        self.i = i

        self.agent = self.performer.agent
        self.environment = self.performer.environment
        self.logger = self.performer.logger

        self.parent = parent
        self.root = self.parent.root if self.parent is not None else self

        self.initVariables()
    
    @classmethod
    def perform(self, performer, path, config, parent=None, i=0, depth=0):
        p = Performance(performer, path, config, parent, i, depth)
        return p.execute(), p
    
    def initVariables(self):
        self.results = []
        self.lastResults = []
        self.lastSequence = []
        self.lastSequenceNode = None
        # print('((START))')
    
        self.absoluteGoal = None
        if self.path.goal:
            self.absoluteGoal = self.path.goal.absoluteData(self.environment.state())
        if self.path.planSettings:
            self.path.planSettings.performing = True
        self.model = self.path.model

        self.replanning = 0
        self.maxReplanning = 5
        self.maxDistance = 7.
        self.maxDerive = None
        if self.model:
            self.maxDistance = self.model.getPrecision(self.MAX_DISTANCE, 1.5)
            self.maxDerive = self.model.getPrecision(self.MAX_DERIVE, 2.5)
        self.nodes = list(self.path.nodes)

        self.formatParameters = FormatParameters()
        self.observationsPrevious = None
        self.distanceToGoal = None
        self.distanceToGoalSinglePass = None
        self.currentPos = None

        self.success = False
        self.fatal = False
        self.topReplanning = False
    
    def execute(self):
        while not self.success and not self.fatal:
            if self.replan():
                break

            if not self.nodes:
                break

            for node in self.nodes:
                self.lastSequenceNode = PerformedSequence(node, self.depth)
                self.root.lastSequence.append(self.lastSequenceNode)

                if self.performNodeContext(node):
                    break

                if self.computePrimitiveForNode(node):
                    break

                if self.checkPreExecution(node):
                    break

                if self.performNode(node):
                    break

                # self.logger.debug(f'Iter (d{self.depth}) {self.i}:\n{node.ty0} Estimated state:\n{node.state.context()}\nCurrent New State:\n{self.environment.state().context()}')
                # self.logger.info(f'Iter (d{self.depth}) {self.i}: performing {node.action}')

                self.processMemory()
                self.checkPostExecution(node)

                if self.checkGoalReached(node):
                    break

                if self.checkDerive(node):
                    break
                self.i += 1
            self.nodes = None
        # if o:
        #     y = o.difference(oStart)
        #results.append(InteractionEvent(self.getIteration(), actionListExecuted, y))

        # print('((OVER))')

        self.postPerformance()

        return self.results, (self.fatal, self.topReplanning)

    def replan(self):
        if not self.nodes and self.absoluteGoal and self.config.allowReplanning:
            if self.distanceToGoal:
                if self.replanning == 0:
                    self.distanceToGoalSinglePass = self.distanceToGoal
                # self.postPerforming(model, False, distanceToGoal)
            self.replanning += 1
            self.config.result.performerReplanning += 1
            if self.replanning > self.maxReplanning:
                self.logger.warning(f'Iter (d{self.depth}) {self.i}: out of replanning')
                return True
            try:
                self.logger.warning(f'Iter (d{self.depth}) {self.i}: no more action to execute... trying to replan new ones')
                newPath, _ = self.agent.planner.plan(self.absoluteGoal, settings=self.path.planSettings)
                if newPath:
                    self.nodes = newPath.nodes
            except ActionNotFound:
                self.logger.warning(f'Iter (d{self.depth}) {self.i}: failed to plan new one')
        return False
    
    def performNodeContext(self, node):
        if node.context:
            subResults, (self.fatal, self.topReplanning) = self.perform(self.performer, node.context, self.config, parent=self, depth=self.depth+1)[0]
            if self.fatal:
                return True
            self.results += subResults
            self.observationsPrevious = None
        return False
    
    def performNode(self, node):
        if node.execution:
            self.observationsPrevious = self.agent.observe(formatParameters=self.formatParameters)
            subResults, (self.fatal, self.topReplanning) = self.perform(self.performer, node.execution, self.config, parent=self, depth=self.depth+1)[0]
            if self.fatal:
                return True
            self.results += subResults
            observations = self.agent.observe(formatParameters=self.formatParameters)
            self.differences = observations.difference(self.observationsPrevious)
            self.observationsPrevious = None
        else:
            # self.logger.debug(f'Iter (d{self.depth}) {self.i}:\n{node.model.npForward(node.action, currentState.context())}')
            event, self.differences, self.observationsPrevious = self.performAction(node, self.observationsPrevious)
            self.results.append(event)
            self.lastResults.append(event)
        return False
    
    def computePrimitiveForNode(self, node):
        if not node.action.space.primitive() and (self.RECOMPUTE_EXECUTION or not node.execution):
            try:
                node.execution, _ = self.agent.planner.plan(node.action, settings=PlanSettings(performing=True))
            except ActionNotFound:
                pass
            if not node.execution:
                self.logger.warning(f'Iter (d{self.depth}) {self.i}: failed to break down non primitive action {node.action}')
                self.fatal = True
                return True
        return False
    
    def postPerforming(self, success, distanceToGoal, singlePrecision=False):
        if self.model:
            self.model.updatePrecision(success, distanceToGoal, index=int(singlePrecision))
    
    def postPerformance(self):
        if self.distanceToGoal:
            if self.replanning == 0:
                self.distanceToGoalSinglePass = self.distanceToGoal
            if self.distanceToGoalSinglePass:
                self.postPerforming(self.success, self.distanceToGoalSinglePass, True)
            self.postPerforming(self.success, self.distanceToGoal * (1 + self.replanning * 0.1))

            if self.depth == 0:
                self.config.result.performerDistance = (self.distanceToGoal, self.maxDistance)
    
    def performAction(self, node, observationsPrevious):
        rawAction = node.action
        action = (node.goal if node and node.goal else rawAction).clone()
        primitiveAction = rawAction

        if observationsPrevious is None:
            observationsPrevious = self.agent.observe(formatParameters=self.formatParameters)
        self.agent._performAction(rawAction, config=self.config)
        observations = self.agent.observe(formatParameters=self.formatParameters)
        differences = observations.difference(observationsPrevious)
        event = InteractionEvent(self.environment.counter.last,
                                 action,
                                 primitiveAction,
                                 differences,
                                 observationsPrevious.convertTo(kind=SpaceKind.PRE))
        observationsPrevious = observations

        return event, differences, observationsPrevious
    
    def processMemory(self):
        if self.agent.learningPolicy == LearningPolicy.EACH_ITERATION:
            self.agent.addMemory(self.lastResults, self.config)
            self.lastResults = []

    def checkGoalReached(self, node):
        if self.absoluteGoal:
            relative = self.absoluteGoal.relativeData(self.currentState)
            self.distanceToGoal = relative.norm()
            if self.distanceToGoal < self.maxDistance and node == self.nodes[-1]:
                self.root.lastSequence.append(PerformedInfo(f'Goal reached {self.distanceToGoal:.3f} < {self.maxDistance:.3f}'))
                self.logger.debug(f'Iter (d{self.depth}) {self.i}: close enough to goal! {self.distanceToGoal:.3f} (max {self.maxDistance:.3f}) ({relative})')
                self.success = True
                return True
            else:
                self.logger.debug(f'Iter (d{self.depth}) {self.i}: distance to goal {self.distanceToGoal:.3f} (max {self.maxDistance:.3f}) ({relative})')
        return False
    
    def checkStatus(self, node, goal, currentState=None):
        if currentState is None:
            currentState = self.environment.state()
        derive = goal.relativeData(currentState, ignoreRelative=True)
        currentPos = goal - derive
        distanceDerive = derive.norm()
        return currentPos, distanceDerive, currentState
    
    def checkPreExecution(self, node):
        if node.absPos:
            self.currentPos, derive, self.currentState = self.checkStatus(node, node.parent.absPos)
            self.logger.debug(
                f'Iter (d{self.depth}) {self.i}: pre execution check: should be at {node.parent.absPos} and currently at {self.currentPos}, ' +
                f'diff {derive:.4f}\nCurrent Pre State:\n{self.currentState.context()}')
            
            self.lastSequenceNode.startingPosition = self.currentPos
            self.lastSequenceNode.startingDerive = derive
            
            if self.model:
                context = self.environment.state().context()
                # self.logger.info(f'Iter (d{self.depth}) {self.i}: {context}')
                predictedOutcome, predictedLinearError = self.model.forward(node.action, context=context)
                predictionDerive = (predictedOutcome - node.goal).norm()

                self.lastSequenceNode.predictedOutcome = predictedOutcome
                self.lastSequenceNode.startingPredictionDerive = predictionDerive

                # if predictionDerive > self.maxDerive * .5 or 
                if predictedLinearError > self.MAX_PREDICTED_LINEAR_ERROR:
                    self.root.lastSequence.append(PerformedInfo(
                        f'Replanning because of exceeding prediction derive {predictionDerive:.4f} > {self.maxDerive:.4f} ({predictedLinearError:.6f})'))
                    self.logger.info(
                        f'Iter (d{self.depth}) {self.i}: max prediction derive exceeded ({predictionDerive:.4f} > {self.maxDerive:.4f}) trying to reach {node.goal}')

                    if self.replanning <= self.maxReplanning:
                        self.logger.info(f'Replanning...')
                        self.nodes = None
                        return True
                    else:
                        self.logger.warning(f'No replanning left!')

                    self.root.lastSequence.append(PerformedInfo(f'Replanning because of exceeding prediction derive {predictionDerive:.4f} > {self.maxDerive:.4f}'))
        return False
    
    def checkPostExecution(self, node):
        self.currentState = self.environment.state()
        # self.logger.warning(currentState)
        self.relativeDifference, self.relativeDerive = None, -1
        if node.absPos:
            self.currentPos, derive, self.currentState = self.checkStatus(
                node, node.absPos, self.currentState)

            # if differences is not None else None
            self.relativeDifference = self.differences.projection(node.absPos.space)
            # if differences is not None else -1.
            self.relativeDerive = (self.relativeDifference - node.goal).norm()
            if self.model:
                self.model.updatePrecisionPerGoal(node.goal, self.relativeDerive)
            self.logger.debug(
                f'Iter (d{self.depth}) {self.i}: wanting {node.absPos} and got {self.currentPos} \n     Diff {derive:.4f} doing {node.action} to get {node.goal} ' +
                f'and got {self.relativeDifference} Diff {self.relativeDerive:.4f}')
            
            if self.model:
                self.lastSequenceNode.endingPredictionDerive = (self.lastSequenceNode.predictedOutcome - self.relativeDifference).norm()
            
            self.lastSequenceNode.endingPosition = self.currentPos
            self.lastSequenceNode.endingDerive = self.relativeDerive

    
    def checkDerive(self, node):
        if node.absPos and self.maxDerive:
            self.currentPos, derive, self.currentState = self.checkStatus(node, node.absPos, self.currentState)
            # print(node.action)
            # print(derive, maxDerive)
            if derive > self.maxDerive:
                if self.model:
                    predictedOutcome, _ = self.model.forward(node.action, context=self.environment.state().context())
                    self.logger.info(f'Iter (d{self.depth}) {self.i}: prediction is {predictedOutcome} ?= planned {node.goal} ?= got {self.relativeDifference}')
                self.root.lastSequence.append(PerformedInfo(f'Replanning because of exceeding derive {derive:.4f} > {self.maxDerive:.4f}'))
                self.logger.info(
                    f'Iter (d{self.depth}) {self.i}: max derive exceeded ({derive:.4f} > {self.maxDerive:.4f}) trying to reach {node.goal} by doing ' +
                    f'{node.action} to get {node.goal} and got {self.relativeDifference} Diff {self.relativeDerive:.4f}')

                self.postPerforming(False, derive)
                self.postPerforming(False, derive, True)

                if self.depth == 0:
                    self.config.result.performerDerive.append((derive, self.maxDerive))
                if self.replanning <= self.maxReplanning:
                    self.logger.info(f'Replanning...')
                    self.nodes = None
                    return True
                else:
                    self.logger.warning(f'No replanning left!')
            # print(f'Derive is {derive} {node.absPos} {node.absPos.relativeData(self.environment.state(), ignoreRelative=True)}')
            # print(self.environment.state().context())
            # print(f'Max {self.MAX_DERIVE * node.absPos.space.maxDistance}')
        return False


