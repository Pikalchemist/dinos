'''
    File name: environment.py
    Author: Alexandre Manoury
    Python Version: 3.6
'''

import sys
import time
import copy
import random
import numpy as np
import threading

from exlab.lab.counter import IterationCounter
from exlab.modular.module import manage
from exlab.interface.graph import Graph

from dino.utils.move import MoveConfig
from dino.data.space import Space
from dino.data.spacemanager import SpaceManager

from dino.evaluation.evaluator import Evaluator

from dino.representation.live_entity import LiveEntity
from dino.representation.property import Property
from dino.representation.state import State

from .engines.engine import Engine
from .scene import SceneSetup

# from .entity import Entity, PhysicalEntity
# from .property import Property
# from ..utils.logging import Logger
# from ..utils.serializer import Serializer


# def make(id_, scene=None, params={}):
#     from ..utils.loaders import DataManager
#     return DataManager.makeEnv(id_, scene, params)


class Environment(SpaceManager):
    """
    Describes the environment for the agent in an experiment.
    Root entity of a world
    """

    DESCRIPTION = ''
    VARYING = False
    CLASS_ENDNAME = 'Environment'

    ENGINE = Engine

    sceneClasses = []
    defaultSceneCls = None

    def __init__(self, sceneCls=None, options={}):
        """
        """
        SpaceManager.__init__(self, storesData=False, entityCls=LiveEntity)
        # self.name = options.get('name', 'Environment')

        assert(self.__class__.ENGINE is not None)
        self.engine = self.__class__.ENGINE(self, options.get('engine', {}))

        manage(self).attach_counter(IterationCounter())
        self.scheduledActionCounter = 0
        self.lock = threading.Lock()
        self.terminateThreads = False

        self.evaluators = []

        # Configuration
        self.options = options
        # self.discrete = options.get('discrete', False)

        self.threading = True
        self.timestep = options.get('timestep', 3.0)  # 2 seconds per action
        # self.unitWindow = options.get("window", 5)  # number of unit

        self.timeByIteration = []

        # Entities
        self.physicalObjects = []

        # Scenes
        self.scene = None
        self.defaultScene = self.defaultSceneCls

        if isinstance(sceneCls, SceneSetup):
            self.registerScene(sceneCls, default=True)
        elif isinstance(sceneCls, str):
            self.defaultScene = self.findScene(sceneCls)
        elif 'scene' in options:
            self.defaultScene = self.findScene(options['scene'])

        if self.defaultScene:
            self.setupScene(self.defaultScene)
    
    @property
    def counter(self):
        return manage(self).counter
    
    @property
    def iteration(self):
        return manage(self).counter.t

    def describe(self):
        spacesDescription = [f"{space}: {space.dim} dimension(s)" for space in self.spaces]
        spaces = '\n'.join(spacesDescription)
        return f"Environment '{self.__class__.__name__}':\n{self.__class__.DESCRIPTION}\n\nSpaces available:\n{spaces}"

    # def setup(self, dataset=None):
    #     # self.bindSpaces()
    #     self.computeSpaces()
    #     if dataset:
    #         for space in self.actionExplorationSpaces:
    #             self.dataset.convertSpace(space)

    @property
    def name(self):
        name = self.__class__.__name__
        if name.endswith(self.CLASS_ENDNAME):
            name = name[:-len(self.CLASS_ENDNAME)]
        return name

    # Scenes
    @classmethod
    def registerScene(cls, sceneCls, default=False):
        if default or not cls.sceneClasses:
            cls.defaultSceneCls = sceneCls
        if sceneCls not in cls.sceneClasses:
            cls.sceneClasses.append(sceneCls)

    # def addSceneAndSetup(self, scene, overwrite=True):
    #     self.addScene(scene)
    #     if overwrite or not self.scene:
    #         self.setupScene(scene)

    def setupScene(self, sceneCls):
        if isinstance(sceneCls, str):
            sceneCls = self.findScene(sceneCls)
        if sceneCls not in self.sceneClasses:
            self.registerScene(sceneCls)
        if self.scene is not None:
            self.clear()
        self.scene = sceneCls(self)
        self.scene.setup()
        self.world.activate()
        self.scene.setupTests()

    def findScene(self, nameCls):
        sceneCls = next(
            (s for s in self.sceneClasses if s.__name__ == nameCls), None)
        if sceneCls is None:
            raise Exception(
                f'Scene named \'{nameCls}\' not found for environment {self}')
        return sceneCls

    @property
    def tests(self):
        return self.scene.tests

    def setupIteration(self, config=MoveConfig()):
        with self.lock:
            self.scene.setupIteration(config)

    def setupEpisode(self, config=MoveConfig()):
        with self.lock:
            self.scene.setupEpisode(config)

    def setupPreTest(self, test=None):
        with self.lock:
            self.scene.setupPreTest(test)

    def state(self, dataset=None):
        return State(self, self.world.observe().flat(), dataset=dataset)
    
    # Evaluation
    def evaluator(self, agent, create=False):
        evaluator = next((eva for eva in self.evaluators if eva.agent == agent), None)
        if not evaluator and create:
            evaluator = Evaluator(agent, self)
            self.evaluators.append(evaluator)
        return evaluator

    def setupEvaluators(self):
        for agent in self.agents():
            self.evaluator(agent, create=True)
    
    def evaluate(self):
        for agent in self.agents():
            self.evaluator(agent, create=True).evaluate()

    # Entities
    def agents(self):
        return [host.hosting for host in self.world.hosts() if host.hosting]
    
    def scheduledActions(self, notNone=True):
        if notNone:
            return {agent: agent.scheduled for agent in self.agents() if agent.scheduled}
        return {agent: agent.scheduled for agent in self.agents()}
    
    def countScheduledActions(self):
        return len(self.scheduledActions())

    def clear(self):
        self.world.clearChildren()

    def reset(self):
        self._reset()
        if self.scene:
            self.scene._reset()

    def _reset(self):
        pass

    def hardReset(self):
        self.clear()
        if self.scene:
            self.scene.setup()

    # def removeObject(self, obj):
    #     if obj in self.objs:
    #         self.removeEntity(obj)
    #         self.objs.remove(obj)
    #         obj.env = None
    #         self.removePhysics(obj)

    def save(self):
        # print("Saved")
        pass

    def execute(self, action, actionParameters=[], config=None, agent=None, sync=False):
        if isinstance(action, Space):
            action = action.point(actionParameters)
        elif isinstance(action, Property):
            action = action.space.point(actionParameters)

        with self.lock:
            for p in action.flat():
                effector = p.space.boundProperty
                if effector is None or not effector.controllable():
                    raise Exception(
                        f'{p.space} is not bound to an effector!')
                effector.perform(p)
        
            if self.threading:
                self.scheduledActionCounter += 1
        if sync and self.threading:
            self.waitNextIteration(agent)
        else:
            self.run()

        return self.reward(action)
    
    def waitNextIteration(self, agent):
        # iteration = self.counter.t
        # while self.counter.t == iteration:
        if self.terminateThreads:
            raise Exception('ThreadTerminated')
        agent.iterationEvent.wait()
        agent.iterationEvent.clear()
        if self.terminateThreads:
            raise Exception('ThreadTerminated')
    
    def checkTerminated(self):
        if self.terminateThreads:
            raise Exception('ThreadTerminated')

    def waitAllScheduledActionsExecuted(self):
        with self.lock:
            awaitedActions = self.countScheduledActions()
        # print(awaitedActions, self.scheduledActionCounter)
        while self.scheduledActionCounter < awaitedActions:
            time.sleep(0.001)
            with self.lock:
                awaitedActions = self.countScheduledActions()
            # print(awaitedActions, self.scheduledActionCounter)
        return awaitedActions > 0

    def step(self, action, actionParameters=[], config=None):
        reward = self.execute(
            action, actionParameters=actionParameters, config=config)
        return self.world.observe(), reward, self.done()

    def _preIteration(self):
        self.scene._preIteration()

    def reward(self, action):
        self.scene.reward(action)

    def done(self):
        return False

    def run(self, evaluating=False):
        # Will run as long as actions are scheduled
        if self.threading:
            for agent, _ in self.scheduledActions().items():
                agent.syncCounter()
            threads = [threading.Thread(target=m) for m in self.scheduledActions().values()]
            for t in threads:
                t.start()

        # Will count as 1 iteration, even if the duration is different from the timestep setting
        lastTime = time.time()
        try:
            while True:
                if self.threading and not self.waitAllScheduledActionsExecuted():
                    return
                with self.lock:
                    self._preIteration()
                    self.engine.run()
                    for host in self.world.hosts():
                        host.scheduledAction = False

                    if self.threading:
                        self.scheduledActionCounter = 0
                    if not evaluating:
                        manage(self).counter.next_iteration()
                    if self.threading:
                        for agent in self.agents():
                            agent.iterationEvent.set()
                if not evaluating:
                    self.timeByIteration.append(time.time() - lastTime)
                    lastTime = time.time()
                if not self.threading:
                    return
        except KeyboardInterrupt:
            if self.threading:
                self.logger.error('KeyboardInterrupt! Killing all threads')
                self.terminateThreads = True
                for agent in self.agents():
                    agent.iterationEvent.set()
                for t in threads:
                    t.join()
                self.terminateThreads = False
                self.logger.info('All threads terminated')

    # Wrappers
    def image(self):
        return self.engine.image()
    
    def show(self):
        self.engine.show()

    def hide(self):
        self.engine.hide()
    
    @property
    def gui(self):
        return self.engine.gui
    
    def displayGui(self, gui=True):
        self.engine.displayGui(gui)
    
    # Visual
    def visualizeTimeByIteration(self, options=None):
        return Graph(options=options).plot(self.timeByIteration)
    
    def visualizeEvaluations(self, options=None):
        g = Graph(options=options)
        for eva in self.evaluators:
            g += eva.visualizeEvaluations()
        return g

    # def _serialize(self, options):
    #     dict_ = SpaceManager._serialize(self, options)
    #     dict_.update(Entity._serialize(self, options))
    #     dict_.update(Serializer.serialize(
    #         self, ['options'], options=options))  # 'discrete',
    #     dict_.update({'scene': self.scene.serialize(options)['id']})
    #     return dict_

    # @classmethod
    # def _deserialize(cls, dict_, options={}, obj=None):
    #     obj = obj if obj else cls(
    #         dict_.get('scene'), options=dict_.get('options', {}))

    #     # SpaceManager._deserialize(dict_, options=options, obj=obj)
    #     Entity._deserialize(dict_, options=options, obj=obj)
    #     return obj

    # @classmethod
    # def deserialize(cls, dict_, options={}, obj=None):
    #     obj = obj if obj else cls(options=dict_)
    #     # spaces = [EnvSpace.deserialize(space, obj) for space in dict_.get('spaces', [])]
    #
    #     # Operations
    #     obj.setup()
    #     obj.testbenchs = dict_['testbench']
    #     return obj

    # def bindSpaces(self):
    #     for property in self.properties():
    #         property.bindSpace(self.spaces)

    # def generateTestbench(self):
    #     testbench = {}
    #     number = 20
    #     print(self.testbenchs)
    #     for tbconfig in self.testbenchs.get('spaces', []):
    #         print(self.space)
    #         space = SpaceManager.space(self, tbconfig['space'])

    #         tbSpace = []

    #         '''for v in np.linspace(minb, maxb, num=number):
    #         point = []
    #         for d, minb, maxb in enumerate(zip(space.bounds['min'], space.bounds['max'])):
    #             for v in np.linspace(minb, maxb, num=number):
    #                 point.append(v)'''
    #         # TODO  id:9
    #         '''tbSpace = np.zeros((space.dim ** number, space.dim))
    #         for d, minb, maxb in enumerate(zip(space.bounds['min'], space.bounds['max'])):
    #             for x in np.linspace(minb, maxb, num=number):
    #                 tbSpace[]

    #         def rec(space, depth=0):
    #             data = []
    #             for x in np.linspace(-50, 50, num=20):
    #                 data.append = np.array([x])
    #             if depth >= space.dim:'''
    #         N = 5
    #         if space.dim == 1:
    #             for x in np.linspace(tbconfig['bounds']['min'][0], tbconfig['bounds']['max'][0], num=N):
    #                 tbSpace.append(np.array([x]))
    #         elif space.dim == 2:
    #             for x in np.linspace(tbconfig['bounds']['min'][0], tbconfig['bounds']['max'][0], num=N):
    #                 for y in np.linspace(tbconfig['bounds']['min'][1], tbconfig['bounds']['max'][1], num=N):
    #                     tbSpace.append(np.array([x, y]))
    #             '''for x in np.linspace(20, 70, num=20):
    #                 for y in np.linspace(-10, 10, num=20):
    #                     tbSpace.append(np.array([x, y]))'''
    #         testbench[space.name] = tbSpace
    #     # print(testbench.keys())
    #     # print("Generated testbench for {}".format(', '.join(testbench.keys())))
    #     Logger.main().info("Testbench generated for {}".format(
    #         ', '.join(list(testbench.keys()))))
    #     return testbench
