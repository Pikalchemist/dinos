from dino.utils.move import MoveConfig


class Challenge(object):
    def __init__(self, method, name=None):
        self.method = method
        self.name = name if name else method.__name__
        self.scene = None

    def __repr__(self):
        return f'Challenge {self.name}'

    def attempt(self, agent, video=False):
        self.agent = agent
        if video:
            self.world.record()
        self.method(self)
        if video:
            self.world.record(False)
            return self.world.video()

    @property
    def world(self):
        return self.scene.world

    def reach(self, goal, config=MoveConfig()):
        origin = self.world.observe(spaces=goal.space.flatSpaces)
        self.agent.reachGoal(goal, config)
        final = self.world.observe(spaces=goal.space.flatSpaces)
        print(f'{self} result: reached {final}, asked {goal} (from {origin})')


class SceneSetup(object):
    CLASS_ENDNAME = 'Scene'

    def __init__(self, environment):
        self.environment = environment

        self.challenges = []
        self.tests = []
        self.testIds = {}
        self._configure()
    
    @property
    def world(self):
        return self.environment.world

    def serialize(self, options={}):
        dict_ = {'id': self.__class__.__name__}
        return dict_
    
    @property
    def name(self):
        name = self.__class__.__name__
        if name.endswith(self.CLASS_ENDNAME):
            name = name[:-len(self.CLASS_ENDNAME)]
        return name

    def __repr__(self):
        return f'SceneSetup {self.__class__.__name__} for env {self.world.name}'

    def _configure(self):
        pass

    def _setup(self):
        pass

    def setup(self):
        self._setup()

    def _setupTests(self):
        pass

    def setupTests(self):
        self._setupTests()

    # Before iteration / episode
    def setupIteration(self, config=MoveConfig()):
        pass

    def setupEpisode(self, config=MoveConfig()):
        pass

    def setupPreTest(self, test=None):
        pass

    def reset(self):
        self._reset()

    def _reset(self):
        pass

    def _draw(self):
        pass

    def _preIteration(self):
        pass

    def reward(self, action):
        return 0.

    def addChallenge(self, challenge):
        if challenge not in self.challenges:
            self.challenges.append(challenge)
            challenge.scene = self

    def addTest(self, test):
        if test not in self.tests:
            self.tests.append(test)
            test.scene = self
            test._bindId()
