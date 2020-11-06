import numpy as np
import math
import copy
import multiprocessing as mp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean

from exlab.interface.serializer import Serializable
from exlab.interface.graph import Graph

from dino.data.data import Goal, SingleData
from dino.data.path import ActionNotFound

from dino.utils.move import MoveConfig
# from dino.utils.maths import iterrange
# from ..utils.io import getVisual, plotData, visualize

from .tests import Test, TestResult


class Evaluation(Serializable):
    def __init__(self, iteration, results, models=[]):
        self.iteration = iteration
        self.results = results
        self.meanError = np.mean([result.meanError for result in self.results])
        self.meanStd = np.mean([result.std for result in self.results])
        self.models = models

    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['iteration', 'meanError', 'meanStd', 'results', 'models'],
                                     exportPathType=True)
        return dict_
    
    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls(dict_.get('iteration'),
                      serializer.deserialize(dict_.get('results', [])))
        return super()._deserialize(dict_, serializer, obj)

    # @classmethod
    # def _deserialize(cls, dict_, options={}, obj=None):
    #     obj = obj if obj else cls(dict_['iteration'],
    #                               dict_['meanError'],
    #                               dict_['meanStd'],
    #                               Serializer.deserialize(
    #                                   dict_['results'], options=options),
    #                               Serializer.deserialize(dict_['models'], options=options))

    #     return obj

    def __repr__(self):
        return f'Evaluation @t={self.iteration} µ={self.meanError:.3f} σ={self.meanStd:.3f} ({len(self.results)} test(s))'


class Evaluator(Serializable):
    """Evaluation of a learning agent on a testbench."""
    DEFAULT_ERROR = 1.0

    def __init__(self, agent, environment, options={}, method=None):
        """
        agent LearningAgent
        """
        self.agent = agent
        self.environment = environment
        self.options = options
        self.method = method if method else options.get('method', 'execution')
        self.ts = options.get('ts', [])
        # Will contain all evaluation performed
        # Each evaluation structured as below:
        #    [t, [[[perf_point0, perf_point1,...], sumError_task_0, std_task_0], ...], errors, stds]
        self.evaluations = {}

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['options', 'method', 'evaluations'], exportPathType=True)
        return dict_
    
    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls(serializer.get('agent'),
                      serializer.get('environment'),
                      options=dict_.get('options', {}),
                      method=dict_.get('method'))

        return super()._deserialize(dict_, serializer, obj)

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        self.evaluations = serializer.deserialize(dict_.get('evaluations'), default={})
        # for evaluation in dict_.get('evaluations', []):
        #     obj = serializer.deserialize(evaluation)
        #     self.evaluations[obj.iteration] = obj

    # @classmethod
    # def _deserialize(cls, dict_, agent, environment, options={}, obj=None):
    #     obj = obj if obj else cls(
    #         agent, environment, dict_.get('options'), dict_.get('method'))

    #     # operations
    #     obj.evaluations = Serializer.deserialize(
    #         dict_.get('evaluations'), options=options)

    #     return obj

    # def nextEvaluationIteration(self, currentIteration):
    #     return next(t for t in self.ts if t > currentIteration)

    # def evaluateAllTime(self):
    #     """Evaluate the learning agent at different times."""
    #     for t in self.ts:
    #         self.evaluateAllTests(t)

    def evaluate(self):
        """Evaluate performance and record it with time given."""

        # Will contain all evaluation information
        results = []

        for test in self.environment.tests:
            result = self.evaluateTest(test)
            if result:
                results.append(result)

        if len(results) == 0:
            return None
        
        models = []

        # if self.agent.dataset:
        #     models = [{'competence': model.competence(), 'model': model}
        #               for model in self.agent.dataset.models]
        # else:
        #     models = []

        evaluation = Evaluation(self.environment.iteration, results, models)
        self.evaluations[evaluation.iteration] = evaluation

        return evaluation

    def evaluateTest(self, test):
        """Compute evaluation for a given task."""
        if test.id is None:
            raise Exception('A test must be assigned to a scene to be evaluated!')
        #assert len(self.testbench[task]) > 0

        # space = self.agent.dataset.space(spacename)
        # dataset = self.agent.dataset
        # planner = self.agent.planner
        # performer = self.agent.performer
        #model = self.agent.dataset.findModelByOutcomeSpace(space)

        #if not model:
        #    return

        # Find the size of the task space at the time
        '''exeId = self.agent.epmem_length[t]
        max_i = len(self.agent.dataset.outcomeSpaces[task].data)
        for i in range(len(self.agent.dataset.outcomeSpaces[task].data)):
            if self.agent.dataset.outcomeSpaces[task].ids[i] > exeId:
                max_i = i
                break

        # Retrieve dataset at the time
        data = np.array(self.agent.dataset.outcomeSpaces[task].data[0:max_i])

        # Create KD-Tree for evaluation
        tree = None
        if len(data) > 0:
            tree = cKDTree(data)'''

        #from evaluation.visualization import Visualizer
        #Visualizer([model.get_competence_visualizer(dataset=testbench), model.get_competence_visualizer()]).plot()
        #Visualizer([model.get_competence_visualizer(method=1), model.get_competence_visualizer()]).plot()
        #raw_input()

        # Will contain all evaluation information
        results = []
        # Error sum on the testbench
        # Sum of quadratic error on testbench
        # n_ = 0

        for rawPoint in test.points:
            if self.agent.dataset:
                point = self.agent.dataset.convertData(rawPoint)
            else:
                point = rawPoint

            # print("Goal: {} {}".format(point, space))
            # -- Use distance --
            if self.method == 'distance':
                result = self.evaluateDistance(point)
            # -- Use planning (and perform the planned path) --
            elif self.method == 'planning':
                self.environment.setupPreTest(test)
                result = self.evaluatePlanning(point)
            elif self.method == 'execution':
                self.environment.setupPreTest(test)
                result = self.evaluateExecution(point)
            '''if tree is None:
                dist = self.agent.dataset.outcomeSpaces[task].options['out_dist']
            else:
                dist, _ = tree.query(point, k=1, eps=0.0, p=2)'''
            #p = std / space.options['max_dist']
            #if p > 10.:
            #    print("{} -> {} {}".format(point, p, competence))
            error = result[0]
            if error >= 0.:
                results.append(result)

        #print(eva)
        # errors.sort()
        # we delete aberations
        #n = 10  # len(eva) / 10 + 1
        #if eva > n:
        #    eva = eva[0:-n]

        return TestResult(test, self.agent.iteration, results, method=self.method)

    def evaluateDistance(self, point):
        _, dist = point.space.nearestDistance(point)
        if not dist:
            return Evaluator.DEFAULT_ERROR, point, None
        return min(1., dist[0] / point.space.maxDistance), point, None

    def evaluatePlanning(self, point):
        try:
            _, _, distance = self.agent.planner.planDistance(
                point)  # , hierarchical=False)
            # print(dist)
            error = min(1., distance / point.space.maxDistance)
        except ActionNotFound as e:
            print(f"Failed {point}")
            error = min(1., e.minDistanceReached /
                        point.space.maxDistance) if e.minDistanceReached else self.DEFAULT_ERROR

            #p2, dist2 = space.nearestDistance(point)
            # print("{} {} // {}".format(point, dist, paths))
            '''if dist < space.options['max_dist'] / 10:
                #print(space.options['max_dist'] / 10)
                #print("Test")
                n_ += 1
                paths = p#planner.plan(goal)
                #print(planner.hierarchical)
                #print(planner.chaining)
                #print(len(paths.paths[0]))
                competence, _, dist = dataset.get_competence_std_paths(paths)#model.get_competence_std(point)
                competence = 0.5 + competence / 2.
                #print(competence)
            else:
                competence, dist = (0, dist)#space.options['out_dist'])
                competence = 0.5 - min(1., dist / space.options['max_dist']) / 2.'''

            #error = 0.4 + 0.5 * min(1., dist / space.options['max_dist'])
        return error, point, None

    def evaluateExecution(self, point):
        try:
            point.space._validate()
            discrete = False
            if self.environment.world.discretizeActions:
                print('Wesh!')
                point = Goal(point)
                point.setRelative(False)
                print(point)
                addDistance = 100.
            else:
                addDistance = point.space.maxDistance

            discrete = self.environment.world.discretizeStates
            self.environment.world.discretizeStates = False
            goalDistance = point.relativeData(self.environment.state()).norm()
            absoluteGoal = point.absoluteData(self.environment.state())
            self.environment.world.discretizeStates = discrete

            print('@@@')
            print(absoluteGoal)
            # print(point)
            # print(self.environment.state())

            self.agent.reach(MoveConfig(
                goal=point, evaluating=True))
            self.environment.runScheduled(evaluating=True)

            self.environment.world.discretizeStates = False
            difference = absoluteGoal.relativeData(self.environment.state())
            # print(self.environment.state())
            self.environment.world.discretizeStates = discrete

            reached = absoluteGoal - difference
            distance = difference.norm()
            try:
                error = min(1., distance / min(goalDistance, addDistance))
            except Exception:
                error = 1.
            print('DISTANCE', reached, difference, distance, error)
            return error, point, reached
            # paths, _, dist = self.agent.planner.planDistance(point)#, hierarchical=False)
            # spaces = list(env.dataset.outcomeSpaces)
            # for s in spaces:
            #     env.scene.property(s.property).save()
            # results = self.agent.performer.perform(paths, self.agent.dataset)
        except ActionNotFound:
            return Evaluator.DEFAULT_ERROR, point, None
    
    # Visual
    def visualizeEvaluations(self):
        g = Graph()
        g.plot(list(self.evaluations.keys()),
               list(map(lambda x: x.meanError, self.evaluations.values())),
               std=list(map(lambda x: x.meanStd, self.evaluations.values())))
        return g

    # Plots
    # def plot(self):
    #     """Plot evaluation on all iterations."""
    #     x = np.array([eval.iteration for eval in self.evaluations])
    #     error = np.array([eval.meanError for eval in self.evaluations])
    #     std = np.array([eval.meanStd for eval in self.evaluations])
    #
    #     ax.plot(x, y) #, marker=options['marker'], linestyle=options['linestyle'], color=options['color']
    #     # plt.yscale('log')
    #     plt.grid(True)

    # def plot_mean_task(self, task, ax, options):
    #     """Plot evaluation for a specific task space."""
    #     x = []
    #     y = []
    #     for i in self.evaluation:
    #         x.append(i[0])
    #         y.append(i[1][task][1])
    #     ax.plot(np.array(x), np.array(
    #         y), marker=options['marker'], linestyle=options['linestyle'], color=options['color'])
    #     plt.yscale('log')
    #     plt.grid(True)

    # #### The following functions are here ro ease the use of Visualizers

    # def get_complete_visualizer(self, prefix=""):
    #     """Create dictionary to visualize evaluation."""
    #     dico = {}
    #     dico['limits'] = {'min': [None, None], 'max': [None, None]}
    #     dico['title'] = prefix + "Evaluations of the algo"
    #     dico['color'] = ['k']
    #     dico['marker'] = ['']
    #     dico['linestyle'] = ['solid']
    #     dico['axes'] = ['Iterations', 'Mean error on testbench']
    #     dico['legend'] = []
    #     dico['plots'] = [lambda fig, ax, options: self.plot(ax, options)]

    #     return dico

    # def get_task_visualizer(self, task, prefix=""):
    #     """Create dictionary to visualize evaluation of a specific task space."""
    #     dico = {}
    #     dico['limits'] = {'min': [None, None], 'max': [None, None]}
    #     dico['title'] = prefix + "Evaluations of the algo on task " + str(task)
    #     dico['color'] = ['k']
    #     dico['marker'] = ['']
    #     dico['linestyle'] = ['solid']
    #     dico['axes'] = ['Iterations', 'Mean error on testbench']
    #     dico['legend'] = []
    #     dico['plots'] = [lambda fig, ax,
    #                      options: self.plot_mean_task(task, ax, options)]

    #     return dico

    # # Visual
    # def visualizeEvaluation(self, byTest=True, prefix=''):
    #     """Return a dictionary used to visualize outcomes reached for the specified outcome space."""

    #     """Plot evaluation on all iterations."""
    #     # x = np.array([eval.iteration for eval in self.evaluations])
    #     # error = np.array([eval.meanError for eval in self.evaluations])
    #     # std = np.array([eval.meanStd for eval in self.evaluations])
    #     #
    #     # ax.plot(x, y) #, marker=options['marker'], linestyle=options['linestyle'], color=options['color']
    #     # # plt.yscale('log')
    #     # plt.grid(True)

    #     if byTest:
    #         data = np.array([[(eval.iteration, 1. - result.meanError) for result in eval.results]
    #                          for eval in self.evaluations]).transpose(1, 0, 2)
    #         stds = np.array([[result.std for result in eval.results]
    #                          for eval in self.evaluations]).transpose()
    #     else:
    #         data = [np.array([(eval.iteration, 1. - eval.meanError)
    #                           for eval in self.evaluations])]
    #         stds = [np.array([eval.meanStd for eval in self.evaluations])]

    #     def visual(d, std):
    #         return lambda fig, ax, options: plotData(d, fig, ax, options, std=std, lines=True)

    #     return getVisual(
    #         [visual(d, std) for d, std in zip(data, stds)],
    #         # marker='o',
    #         title=prefix + "Global competences"
    #     )

    # def visualizeCompetence(self, prefix=""):
    #     """Return a dictionary used to visualize outcomes reached for the specified outcome space."""
    #     data = {}
    #     for v in self.evaluation:
    #         for model, mdict in v['models'].items():
    #             if model not in data:
    #                 data[model] = []
    #             data[model].append((v['iteration'], mdict['competence']))
    #     plots = [lambda fig, ax, options: plotData(
    #         np.array(d), fig, ax, options, lines=True) for _, d in data.items()]
    #     legend = [str(model) for model, _ in data.items()]
    #     return getVisual(
    #         plots,
    #         marker='.',
    #         legend=legend,
    #         title=prefix + "Models competence"
    #     )

    # # Plot
    # def plot(self, byTest=True):
    #     visualize(self.visualizeEvaluation(byTest=byTest))

    # def plotCompetence(self):
    #     visualize(self.visualizeCompetence())

    # # Api
    # def apiget_evaluation(self, range_=(-1, -1)):
    #     return {'data': iterrange(self.evaluation, range_)}
