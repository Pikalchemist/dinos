# Dino

Dino is a simulation environment for active learning algorithms.


# Getting started

First of all, install the package either using pip:

    pip install dino

Or from the git repository:

    pip install -r ./requirements.txt
    pip install -e .

Examples are provided in the `examples` folder from the git repository.


# How does it works

To run a Dino experiment you need an `Environment` and an `Agent`.  
For instance, an environment may be initialized as follow:

    from dino.environments.playground import PlaygroundEnvironment
    env = PlaygroundEnvironment()

From there you can either use your own code and use low level API to interact with the environment: `env.step(self, action, actionParameters=[], config=None)` as detailed later on. The second option is to use the Dino Agent system to manage your algorithm.

For instance to create an agent that will perform a random action at each step:

    from dino.agents.random import RandomAgent
    agent = RandomAgent(env.world.findHost())

> `env.world.findHost()` let you find an entity in the environment that can be controlled by your learner (we call such entity an *host*)

Each `Agent` has a `reach(self, configOrGoal)` method that can be used to tell the agent to reach a specific goal.

Additionally a specific type of agent exists: `Learner`. This class is designed to be used with a dataset or a memory to learn from its interactions with the environment.

Each `Learner` has a `train(self, iterations=None, untilIteration=None, episodes=None, untilEpisode=None)` method used to train your learner for a given number of iterations or episodes.

More details are present in the `examples` folder from the git repository.