import numpy as np
import logging

from exlab.interface.serializer import Serializable


class Test(Serializable):
    def __init__(self, space, name='', id=None):
        self.space = space
        self.name = name
        self.id = id
        self.scene = None
        self.points = []

    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['space', 'name', 'id', 'scene', 'points'],
                                     exportPathType=True)
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, options={}, obj=None):
    #     obj = obj if obj else cls(Serializer.deserialize(dict_['space'], options=options),
    #                               dict_.get('name'),  dict_.get('id'))

    #     # operations
    #     obj.points = Serializer.deserialize(
    #         dict_.get('points'), options=options)

    #     return obj

    def _bindId(self):
        assert self.scene is not None
        index = 0

        def name(index):
            return '{}-{}'.format(self.space.boundProperty.reference(), index)

        while name(index) in self.scene.testIds:
            index += 1
        self.id = name(index)
        self.scene.testIds[self.id] = self

    def __repr__(self):
        return 'Test on {} ({} point(s))'.format(self.space, len(self.points))


class TestResult(Serializable):
    def __init__(self, test, iteration, results, method=''):
        self.test = test
        self.iteration = iteration
        print(results)
        # [(error, goal, reached)]
        self.results = results
        self.method = method
        errorsArray = np.array([r[0] for r in results])
        self.meanError = np.mean(errorsArray)
        self.meanQuadError = np.mean(errorsArray ** 2)
        self.std = np.sqrt(self.meanQuadError - self.meanError ** 2)

        # logging.info("Error: {} {} {} [Sum, Quad, Std]".format(
        #     meanError, meanQuadError, std))

    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['test', 'iteration', 'meanError', 'meanQuadError', 'std', 'results',
                                            'method'], exportPathType=True)
        return dict_

    # @classmethod
    # def _deserialize(cls, dict_, options={}, obj=None):
    #     obj = obj if obj else cls(Serializer.deserialize(dict_['test'], options=options),
    #                               dict_['iteration'],
    #                               dict_['meanError'],
    #                               dict_['meanQuadError'],
    #                               dict_['std'],
    #                               Serializer.deserialize(
    #                                   dict_['results'], options=options),
    #                               method=dict_['method'])

    #     return obj

    def __repr__(self):
        return 'Result Test {} @t={} µ={} σ={} ({} point(s))'.format(self.test.id, self.iteration, self.meanError,
                                                                     self.std, len(self.results))

    def details(self):
        results = ['-{}: aimed for {} and reached {}, error: {}'.format(i, goal, reached, error)
                   for i, (error, goal, reached) in enumerate(self.results)]
        return '{}:\n{}'.format(self, '\n'.join(results))


class PointTest(Test):
    def __init__(self, space, pointValue, relative=None, name=''):
        super().__init__(space, name=name)
        self.points.append(space.goal(pointValue).setRelative(relative))


class PointsTest(Test):
    def __init__(self, space, pointValueList, relative=None, name=''):
        super().__init__(space, name=name)
        for value in pointValueList:
            self.points.append(space.goal(value).setRelative(relative))


class UniformGridTest(PointsTest):
    def __init__(self, space, boundaries, numberByAxis=4, relative=None, name=''):
        vectors = [np.linspace(bound[0], bound[1], numberByAxis)
                   for bound in boundaries]
        mesh = np.meshgrid(*vectors)
        flatten = [axis.flatten() for axis in mesh]
        pointValueList = np.array(flatten).T

        super().__init__(space, pointValueList=pointValueList, relative=relative, name=name)
