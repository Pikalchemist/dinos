import copy


class Result(object):
    def __init__(self, config):
        self.config = config

        self.reachedGoal = None
        self.reachedContext = None
        self.action = None

        self.randomProbability = None
        self.planningSuccess = None
        self.planningDistance = None
        self.planningSteps = None
        self.planningChangeContextSteps = []
        self.performerReplanning = 0
        self.performerReplanningSteps = 0
        self.performerDistance = None
        self.performerDerive = []

    def clone(self, config):
        new = copy.copy(self)
        new.config = config
        return new
    
    def results(self):
        score = 0.
        txt = ''

        if self.config.goal:
            txt += f'goal {self.config.goal} '
            if not self.reachedGoal:
                txt += f'not attempted'
            elif isinstance(self.reachedGoal, str):
                txt += f'{self.reachedGoal}'
            else:
                difference = (self.config.goal - self.reachedGoal).norm()
                score += difference / self.config.goal.space.maxDistance * 5.
                txt += f'and got {self.reachedGoal}, difference is {difference}'
            txt += f'|   '

        if self.config.goalContext:
            txt += f'context {self.config.goalContext} '
            if not self.reachedContext:
                txt += f'not attempted'
            elif isinstance(self.reachedContext, str):
                txt += f'{self.reachedContext}'
            else:
                difference = (self.config.goalContext - self.reachedContext).norm()
                score += difference / self.config.goalContext.space.maxDistance * 5.
                txt += f'and got {self.reachedContext}, difference is {difference}'
            txt += f'|   '

        valid = 'Ok' if score < 0.1 else 'Error'
        return f'{valid}: {score} ({txt})'

    def __repr__(self):
        attrResults = ['action', 'reachedGoal', 'reachedContext', 'randomProbability',
                       'planningSuccess', 'planningDistance', 'planningSteps',
                       'planningChangeContextSteps', 'performerReplanning',
                       'performerReplanningSteps', 'performerDistance', 'performerDerive']
        results = ', '.join(
            [f'{k}: {getattr(self, k)}' for k in attrResults if getattr(self, k)])
        return results
