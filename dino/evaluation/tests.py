import numpy as np
import logging

from exlab.interface.serializer import Serializable


class Test(Serializable):
    def __init__(self, name, space, id=None):
        self.space = space
        self.name = name
        self.id = id
        self.scene = None
        self.points = []
    
    def _sid(self, serializer):
        return serializer.uid('test', self.name)

    def _serialize(self, serializer):
        dict_ = serializer.serialize(self, ['space', 'name', 'id', 'scene', 'points'])
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
            return f'{self.space.boundProperty.reference()}-{index}'

        while name(index) in self.scene.testIds:
            index += 1
        self.id = name(index)
        self.scene.testIds[self.id] = self

    def __repr__(self):
        return f'Test on {self.space} ({len(self.points)} point(s))'


class TestResult(Serializable):
    def __init__(self, test, iteration, results, method=''):
        self.test = test
        self.iteration = iteration
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
        dict_ = serializer.serialize(self, ['iteration', 'meanError', 'meanQuadError', 'std', 'results', 'method'],
                                     foreigns=['test'])
        return dict_
    
    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls(serializer.deserialize(dict_.get('test')),
                      dict_.get('iteration'),
                      dict_.get('results'),
                      dict_.get('method', ''))

        return super()._deserialize(dict_, serializer, obj)

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
        return f'Result Test {self.test.id} @t={self.iteration} µ={self.meanError} σ={self.std} ({len(self.results)} point(s))'

    def details(self):
        results = [f'-{i}: aimed for {goal} and reached {reached}, error: {error}'
                   for i, (error, goal, reached) in enumerate(self.results)]
        results = '\n'.join(results)
        return f'{self}:\n{results}'


class PointTest(Test):
    def __init__(self, name, space, pointValue, relative=None):
        super().__init__(name, space)
        self.points.append(space.goal(pointValue).setRelative(relative))


class PointsTest(Test):
    def __init__(self, name, space, pointValueList, relative=None):
        super().__init__(name, space)
        for value in pointValueList:
            self.points.append(space.goal(value).setRelative(relative))


class UniformGridTest(PointsTest):
    def __init__(self, name, space, boundaries, numberByAxis=4, relative=None):
        vectors = [np.linspace(bound[0], bound[1], numberByAxis)
                   for bound in boundaries]
        mesh = np.meshgrid(*vectors)
        flatten = [axis.flatten() for axis in mesh]
        pointValueList = np.array(flatten).T

        super().__init__(name, space, pointValueList=pointValueList, relative=relative)
