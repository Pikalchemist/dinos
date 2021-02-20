from enum import Enum


class LearningPolicy(Enum):
    DEFAULT = 0
    EACH_EPISODE = 1
    EACH_ITERATION = 2
