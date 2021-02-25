import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import copy
import random
import pickle

from exlab.utils.text import strtab
from exlab.interface.graph import Graph
from exlab.interface.serializer import Serializable

from .space import Space, SpaceKind
from .dataspace import DataSpace
# from ..utils.logging import DataEventHistory, DataEventKind
# from ..utils.io import strtab, getVisual, plotData, visualize
# from dino.utils.maths import uniformSampling, first
from .data import Data


class Point(Serializable):
    MAX_VALUES = 10
    WINDOW_SIZE = 5
    NEAR_CONTEXT_DISTANCE = 0.04
    ZERO_RANGE = 0.001

    def __init__(self, id_, position, context, positionContext, value=None):
        self.id = id_
        self.position = position
        self.context = context
        self.positionContext = positionContext

        self.zero = position.norm() < self.ZERO_RANGE

        self.values = []
        self.progresses = []
        self.evaluation = 0.
        self.cost = 1.

        if value is not None:
            self.addValue(value)

    def __repr__(self):
        return f'Point #{self.id} {self.position} {self.evaluation}'
    
    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['id', 'position', 'context', 'positionContext', 'values', 'progresses', 'evaluation', 'cost'])
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            obj = cls(dict_.get('id'),
                      dict_.get('position'),
                      dict_.get('context'),
                      dict_.get('positionContext'),
                      serializer.get('.region.parent'))
        return super()._deserialize(dict_, serializer, obj)

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        for attr in ['values', 'progresses', 'evaluation', 'cost']:
            setattr(self, attr, dict_.get(attr))
    
    @property
    def lastValue(self):
        return self.values[-1]
    
    def needUpdate(self):
        if len(self.progresses) < 4:
            return True
        if random.uniform(0, 1) < self.evaluation * 10:
            return True
        return False

    def addValue(self, value):
        self.values.append(value)
        self.values = self.values[-self.MAX_VALUES:]
        self.computeEvaluation(self.values, self.WINDOW_SIZE, cost=self.cost)
    
    def computeEvaluation(self, values, windowSize, cost=1.):
        """Compute evaluation of the region for given strategy."""
        if len(values) >= windowSize:
            self.progresses.append(self.meanProgress(values))
            self.progresses = self.progresses[-windowSize:]

        if len(self.progresses) >= 2:
            self.evaluation = np.abs((self.progresses[-1] - self.progresses[0]) / cost)
        elif values:
            self.evaluation = values[-1] / cost
        else:
            self.evaluation = 0.

    def meanProgress(self, values):
        """Compute mean progress according to the evaluation window."""
        return np.mean(values[-self.WINDOW_SIZE:])
    
    def nearContext(self, projectedContext=None):
        if not projectedContext:
            return True
        distance = self.context.distanceTo(projectedContext) / self.context.space.maxDistance
        return distance < self.NEAR_CONTEXT_DISTANCE



class SpaceRegion(Serializable):
    """Implements an evaluation region."""

    REGION_BASED = 0
    REGION_UPDATABLE = 1
    POINT_BASED = 2

    def __init__(self, space, options, bounds=None, parent=None, manager=None, tag='',
                 contextSpace=None, regions=None):
        """
        bounds float list dict: contains min and max boundaries of the region
        options dict: different options used by the evaluation model
        """
        self.explorableSpace = space
        self.explorableContextSpace = space.spaceManager.multiColSpace([space, contextSpace], weight=0.5)
        self.contextSpace = contextSpace

        self.parent = parent
        self._manager = manager

        self.bounds = copy.deepcopy(
            bounds) if bounds else Space.infiniteBounds(self.explorableContextSpace.dim)
        assert(parent is not None or manager is not None)
        self.tag = tag

        self.colsExplorable = self.explorableContextSpace.columnsFor(self.explorableSpace)
        self.colsContext = self.explorableContextSpace.columnsFor(self.contextSpace)

        # Contains the following keys:
        #   'minSurface': the minimum surface of a region to be still splittable
        #   'maxPoints': the maximum of points contained in a region (used to split)
        #   'window': the number of last progress measures to consider for computing region evaluation (to take newer
        #             points only)
        #   'cost': cost of strategy (usually strategies involving a teacher have a higher cost)
        self.options = {
            "cost": 1.0,
            "window": 10,
            "maxAttempts": 20,
            "maxPoints": 40,
            "minSurface": 0.05,
            "pointNumberSplit": 50
        }
        self.options.update(options)

        self.number = 0
        self.points = []
        self.nonZeroPoints = []
        self.progresses = []
        self.evaluation = 0.
        # self.base = Point(None, None, None, None)

        self.leftChild = None
        self.rightChild = None
        self.regions = regions if regions else [self]

        self.splitValue = 0.
        self.splitDim = None

        self.setSplittable()

        if not parent:
            # register history only for the root region
            # self.history = DataEventHistory()
            self.childrenNumber = 0
        else:
            self.lid = self.root().childrenNumber
            self.root().childrenNumber += 1

        self.explorablePreSpace = self.explorableSpace.convertTo(self.dataset, kind=SpaceKind.PRE)

    def __repr__(self):
        cut = strtab(f'{self.splitDim}th d: {self.splitValue:.4f}')
        return f'Region {self.explorableContextSpace} {self.evaluation:.4f}\n    Left: {self.leftChild is not None}\n    <Cut {cut}>\n    Right: {self.rightChild is not None}'

    def _serialize(self, serializer):
        dict_ = serializer.serialize(
            self, ['explorableSpace', 'contextSpace', 'bounds', 'points', 'number', 'progresses', 'evaluation',
                   'leftChild', 'rightChild', 'splitValue', 'splitDim', 'tag'])
        return dict_

    @classmethod
    def _deserialize(cls, dict_, serializer, obj=None):
        if obj is None:
            # leftChild = serializer.deserialize(dict_.get('leftChild'))
            # rightChild = serializer.deserialize(dict_.get('rightChild'))
            obj = cls(serializer.deserialize(dict_.get('explorableSpace')),
                      options=dict_.get('options', {}),
                      bounds=dict_.get('bounds'),
                      parent=serializer.get('.region.parent'),
                      manager=serializer.get('.region.manager'),
                      tag=dict_.get('tag', ''),
                      contextSpace=serializer.deserialize(dict_.get('contextSpace')))
        return super()._deserialize(dict_, serializer, obj)

    def _postDeserialize(self, dict_, serializer):
        super()._postDeserialize(dict_, serializer)
        serializer = serializer.clone(values={
            '.region.parent': self,
            '.region.manager': self.manager,
        })

        for attr in ['number', 'progresses', 'evaluation']:
            setattr(self, attr, dict_.get(attr))
        for attr in ['points']:
            setattr(self, attr, serializer.deserialize(dict_.get(attr)))
        self.nonZeroPoints = [p for p in self.points if not p.zero]

        self.leftChild = serializer.deserialize(dict_.get('leftChild'))
        self.rightChild = serializer.deserialize(dict_.get('rightChild'))

    def root(self):
        root = self
        while root.parent is not None:
            root = root.parent
        return root
    
    @property
    def manager(self):
        return self.root()._manager
    
    @property
    def dataset(self):
        return self.root()._manager.dataset
    
    @property
    def method(self):
        return self.root()._manager.method
    
    def positions(self, withContext=True):
        if withContext:
            return np.array([p.positionContext for p in self.points])
        return np.array([p.position for p in self.points])

    def finiteBounds(self):
        # points = np.array(self.root().getSplitData(withRegions=False)[0])
        positions = self.positions()
        return [(np.min(positions[:, j]) if bmin == -math.inf else bmin,
                 np.max(positions[:, j]) if bmax == math.inf else bmax)
                for j, (bmin, bmax) in enumerate(self.bounds)]

    def maxDistance(self):
        positions = self.positions()
        bounds = np.max(positions, axis=0) - np.min(positions, axis=0)
        return np.sum(bounds ** 2) ** 0.5

    @property
    def splitten(self):
        return self.leftChild is not None

    # def nearestContext(self, context):
    #     self.contextSpace._validate()
    #     context = context.convertTo(kind=SpaceKind.PRE).projection(self.contextSpace).plain()

    #     indices, distances = DataSpace.nearestFromData(np.array(self.points)[:, self.colsContext], context,
    #                                                             n=self.options['maxPoints']//2)

    #     return indices, distances

    def nearContext(self, context, tolerance=0.01):
        if not self.contextSpace:
            return True

        self.contextSpace._validate()
        context = context.convertTo(kind=SpaceKind.PRE).projection(self.contextSpace).plain()
        tolerance = self.contextSpace.maxDistance * tolerance

        # Context in bounds
        inBounds = True
        for i, col in enumerate(self.colsContext):
            if context[i] < self.bounds[col][0] - tolerance or context[i] > self.bounds[col][1] + tolerance:
                inBounds = False
                break
        return inBounds

    def controllableContext(self):
        if not self.contextSpace:
            return False
        return self.dataset.controllableSpaces(self.contextSpace)
    
    def updatePoints(self, aroundPoint, closeIds=None):
        for point in self.nonZeroPoints:
            if point.id in closeIds and point != aroundPoint:
                self.updatePoint(point)

    def updatePoint(self, point):
        pass

    def computeEvaluation(self):
        if self.method == self.POINT_BASED:
            self.evaluation = max(p.evaluation for p in self.points)
        elif self.method == self.REGION_BASED:
            values = [p.lastValue for p in self.points]
            Point.computeEvaluation(self, values, self.options['window'])

    def findRandom(self, allowZeros=False):
        if allowZeros:
            return random.choice(self.points)
        if not self.nonZeroPoints:
            return None
        return random.choice(self.nonZeroPoints)
 
    def findBest(self, context=None, changeContext=True, favoriseBest=1.):
        dataChangeContext, dataCurrentContext = [], []
        if context:
            context = context.projection(self.contextSpace)
            dataCurrentContext += [point for point in self.nonZeroPoints if point.nearContext(context)]

        if changeContext:
            dataChangeContext += list(self.nonZeroPoints)
            changeContext = self.chooseToChangeContext(max(x.evaluation for x in dataChangeContext) if dataChangeContext else -math.inf,
                                                       max(x.evaluation for x in dataCurrentContext) if dataCurrentContext else -math.inf)

        data = dataChangeContext if changeContext else dataCurrentContext
        # data.sort(key=lambda x: -x[1])

        if not data:
            return self.findRandom(), changeContext
        return random.choices(data, weights=[x.evaluation for x in data])[0], changeContext
    
    def createRandomPoint(self):
        return self.explorableSpace.point(np.array([random.uniform(bmin, bmax) for bmin, bmax in self.finiteBounds()])[self.colsExplorable])
    
    def chooseToChangeContext(self, bestChangedContext, bestCurrentContext):
        if bestCurrentContext > bestChangedContext * 4:
            return False
        if bestChangedContext > bestCurrentContext * 4:
            return True
        return random.uniform(0, bestChangedContext + bestCurrentContext) < bestChangedContext
    
    def getGoalContext(self, point):
        if self.controllableContext():
            return point.context
        return self.explorableSpace.point(self.explorablePreSpace.getData([point.id])[0]).setRelative(False)

    def addPoint(self, id_, point, context, value, closeIds=None, firstAlwaysNull=True, populating=False):
        """Add a point and its value in the attached evaluation region."""
        pointContext = Data.npPlainData(point.extends(context), self.explorableContextSpace)
        # point = Data.plainData(point, self.explorableSpace)
        # context = Data.plainData(context, self.contextSpace)
        point = point.projection(self.explorableSpace, kindSensitive=True)
        context = context.projection(self.contextSpace, kindSensitive=True)
        assert len(pointContext) == len(self.bounds)

        # print("ADDING 1 POINT", point)
        # if filterNull and np.all(np.array(point)[self.colsTarget]==0):
        #     print("NULL POINT")
        #     return

        # if firstAlwaysNull and len(self.points) == 0:
        #     value = 0.

        p = Point(id_, point, context, pointContext, value)
        self._addPointData(p, closeIds=closeIds, firstAlwaysNull=firstAlwaysNull, populating=populating)

    def _addPointData(self, point, closeIds=None, firstAlwaysNull=True, populating=False):
        self.points.append(point)
        if not point.zero:
            self.nonZeroPoints.append(point)
        self.number += 1

        if not populating and self.method == self.POINT_BASED:
            self.updatePoints(point, closeIds)
        self.computeEvaluation()

        # addedInChildren = False
        if not self.splitten and self.splittable and self.number > self.options['pointNumberSplit']:
            # Leave reached, region must be splitten
            self.randomCut(self.options['maxAttempts'])
            self.split()
        elif self.splitten:
            # Traverse tree
            # addedInChildren = True
            child = self.leftChild if point.positionContext[self.splitDim] < self.splitValue else self.rightChild
            child._addPointData(point, closeIds=closeIds, populating=populating)

        # Remove oldest points and pointValues
        if self.method == self.POINT_BASED:
            pass
        else:
            self.points = self.points[-self.options['maxPoints']:]

        # if not addedInChildren and not populating and self.manager:
        #     self.manager.logger.debug(f'Adding point [{", ".join(["{:.4f}".format(p) for p in point])}] with value {value:.3e} to region {self}', tag=self.tag)

    # Splitting process
    def setSplittable(self):  # Change it to use max depth tree instead ???
        """Check if the region is splittable according to its surface."""
        surface = 1.
        for bmin, bmax in self.bounds:
            surface *= (bmax - bmin)
        self.splittable = (surface > self.options['minSurface'])

    def split(self):
        """Split region according to the cut decided beforehand."""

        if self.splitDim is not None:
            return
        if self.manager:
            self.manager.logger.debug(f'Splitting along dim {self.splitDim}: {self.splitValue:.4f} for {self}', tag=self.tag)

        # Create child regions boundaries
        leftBounds = copy.deepcopy(self.bounds)
        leftBounds[self.splitDim][1] = self.splitValue
        rightBounds = copy.deepcopy(self.bounds)
        rightBounds[self.splitDim][0] = self.splitValue

        # Create empty child regions
        self.leftChild = self.__class__(self.explorableSpace, self.options, bounds=leftBounds,
                                        parent=self, contextSpace=self.contextSpace, regions=self.regions)
        self.rightChild = self.__class__(self.explorableSpace, self.options, bounds=rightBounds,
                                         parent=self, contextSpace=self.contextSpace, regions=self.regions)
        # root = self.root()
        # root.history.append(root.manager.getIteration(), DataEventKind.ADD, [(str(self.leftChild.lid), self.leftChild.serialize())])
        # root.history.append(root.manager.getIteration(), DataEventKind.ADD, [(str(self.rightChild.lid), self.rightChild.serialize())])

        # Add them in the list of regions (that's the only reason we need access to the list of all regions!!!)
        self.regions.append(self.leftChild)
        self.regions.append(self.rightChild)

        #print("-----------------------------")
        #print("Split done")
        #print("Region [" + str(self.bounds['min']) + ", " + str(self.bounds['max']) + "] ----> Left ["
        #      + str(left_region.bounds['min']) + ", " + str(left_region.bounds['max']) + "] (")

        #left = 0
        #right = 0
        #for s in range(len(self.points)):
        #    for i in range(len(self.points[s])):
        #        point = self.points[s][i]
        #        if point[self.splitDim] < self.splitValue:
        #            left += 1
        #        else:
        #            right += 1

        #print("Split put " + str(left) + " points left, " + str(right) + " points right.")
        #print("Split done on dimension " + str(self.splitDim) + " at value: " + str(self.splitValue))

        # Add all points of the parent region in the child regions according to the cut
        for point in self.points:
            child = self.leftChild if point.positionContext[self.splitDim] < self.splitValue else self.rightChild
            child._addPointData(point, populating=True)

    # def greedyCut(self):
    #     """UNTESTED method to define a cut greedily."""
    #     i = range(len(self.points))
    #     maxQ = 0.

    #     # For each dimension
    #     for d in range(len(self.bounds['min'])):
    #         # Sort the points in the dimension
    #         i.sort(key=lambda j: self.points[j][d])
    #         for p in range(len(i)-1):
    #             progress_left = 0.
    #             progress_right = 0.

    #             left = copy.deepcopy(i[0:(p+1)])
    #             right = copy.deepcopy(i[(p+1):len(i)])

    #             n = min(len(left), self.options['window'])
    #             if n < len(left):
    #                 left.sort()
    #             for k in range(n):
    #                 progress_left += self.pointValues[left[len(left)-k-1]]
    #             progress_left /= float(n)

    #             n = min(len(right), self.options['window'])
    #             if n < len(right):
    #                 right.sort()
    #             for k in range(n):
    #                 progress_right += self.pointValues[right[len(right)-k-1]]
    #             progress_right /= float(n)

    #             delta_p = (progress_right - progress_left)**2
    #             Q = delta_p * len(left) * len(right)

    #             if Q > maxQ:
    #                 # Choose mean between two points as the cut
    #                 self.splitDim = d
    #                 self.splitValue = (
    #                     self.points[i[p]][d] + self.points[i[p+1]][d]) / 2.

    # Careful !!! The tree can't handle it if it has only one value multiple times !!!
    def randomCut(self, numberAttempts):
        """Define the cut by testing a few cuts per dimension."""
        if self.manager:
            self.manager.logger.debug2(f'Trying to cut region {self}', tag=self.tag)
        i = list(range(len(self.points)))
        #n = int(math.ceil(len(self.points)/(numberAttempts+1)))
        n = float(len(self.points)) / float(numberAttempts + 1)
        maxQ = 0.
        #max_card = 0
        #splitValue_card = 0.
        #splitDim_card = -1

        # For each dimension
        #from pprint import pprint
        for d in range(len(self.bounds)):
            i.sort(key=lambda j: self.points[j].positionContext[d])
            # For each attempt
            for k in range(numberAttempts):
                progress_left = 0.
                progress_right = 0.

                # Sort points by age
                #left = copy.deepcopy(i[0:((k+1)*n)])
                #right = copy.deepcopy(i[((k+1)*n):len(i)])
                #left.sort()
                #right.sort()

                # Id of the item following cut
                idcut = int(math.floor((k+1)*n))

                splitValue = (self.points[i[idcut-1]].positionContext[d] + self.points[i[idcut]].positionContext[d]) / 2.

                # Make sure we are not trying to split something unsplittable
                if self.points[i[idcut-1]].positionContext[d] == splitValue:
                    continue

                left = []
                right = []
                for j in range(len(self.points)):
                    if self.points[j].positionContext[d] < splitValue:
                        left.append(j)
                    else:
                        right.append(j)

                # Retain only the points inside evaluation window
                left_filtered = left[(
                    len(left)-min(len(left), self.options['window'])):len(left)]
                right_filtered = right[(
                    len(right)-min(len(right), self.options['window'])):len(right)]

                # Compute progress for each part
                for j in left_filtered:
                    progress_left += self.points[j].lastValue
                progress_left /= float(max(len(left_filtered), 1))
                for j in right_filtered:
                    progress_right += self.points[j].lastValue
                progress_right /= float(max(len(right_filtered), 1))

                # Compute delta_p and Q
                delta_p = (progress_left - progress_right)**2
                card = len(left) * len(right)
                if len(left) > 0 and len(right) > 0:
                    Q = delta_p * float(card)
                else:
                    Q = -float('inf')

                if Q > maxQ:
                    # Choose mean between two points as the cut
                    self.splitDim = d
                    self.splitValue = splitValue
                    maxQ = Q
        if self.manager:
            self.manager.logger.debug2(f'Found split along dim {self.splitDim}: {self.splitValue:.4f} for {self}', tag=self.tag)

    def getSplitData(self, withRegions=True):
        points = self.positions()
        pointValues = [point.value for point in self.points]

        regions = []
        if withRegions and not self.splitten:
            regions.append((self.evaluation, self.finiteBounds()))

        for child in [self.leftChild, self.rightChild]:
            if child:
                p, pv, r = child.getSplitData(withRegions=withRegions)
                points += p
                pointValues += pv
                if withRegions:
                    regions += r
        return points, pointValues, regions

    def _cuts(self):
        if not self.splitten:
            return []
        return [(self.splitDim, self.splitValue)] + self.leftChild._cuts() + self.rightChild._cuts()

    def cuts(self):
        _cuts = self._cuts()
        return _cuts

    # Visual
    def visualizeData(self, options={}, outcomeOnly=True, contextOnly=False, absoluteProgress=False):
        g = Graph(title=f'Regions from {self}', options=options)
        points, pointValues, regions = self.getSplitData()

        points = np.array(points)
        pointValues = np.clip(np.array(pointValues), -100, 100)

        if absoluteProgress:
            pointValues = np.abs(pointValues)

        # Filter
        if contextOnly:
            cols = self.colsContext
            points = points[:, cols]
        elif outcomeOnly:
            cols = self.colsExplorable
            points = points[:, cols]
        else:
            cols = np.arange(self.explorableContextSpace.dim)

        # pvMinimum = np.min(pointValues)
        # pvMaximum = np.max(pointValues)
        evaluation = np.array([r[0] for r in regions])
        evalMinimum = np.min(evaluation)
        # evalMaximum = np.max(evaluation)
        evaluation = (evaluation - evalMinimum) / max(0.001, np.max(evaluation) - evalMinimum)

        for region in regions:
            bounds = np.array(region[1])[cols]
            alpha = (0.2 + region[0]) / 1.2
            if len(bounds) == 2:
                g.rectangle((bounds[0][0], bounds[1][0]), bounds[0][1] - bounds[0]
                            [0], bounds[1][1] - bounds[1][0], alpha=alpha, zorder=-1, border=True)
            else:
                g.rectangle((bounds[0][0], -1), bounds[0][1] -
                            bounds[0][0], 2, alpha=alpha, zorder=-1, border=True)
        g.scatter(points, color=pointValues, colorbar=True)
        return g

    # def getRegionsVisualizer(self, prefix='', outcomeOnly=True, contextOnly=False, absoluteProgress=False):
    #     """Return a dictionary used to visualize evaluation regions."""
    #     points, pointValues, regions = self.getSplitData()

    #     points = np.array(points)
    #     pointValues = np.clip(np.array(pointValues), -100, 100)
    #     if absoluteProgress:
    #         pointValues = np.abs(pointValues)
    #     # regions = np.array(regions)
    #     if contextOnly:
    #         cols = self.colsContext
    #         points = points[:, cols]
    #     elif outcomeOnly:
    #         cols = self.colsTarget
    #         points = points[:, cols]
    #     else:
    #         cols = np.arange(self.space.dim)

    #     pvMinimum = np.min(pointValues)
    #     pvMaximum = np.max(pointValues)
    #     # pointValues = (pointValues - pvMinimum) / max(0.001, pvMaximum - pvMinimum)
    #     # print(pointValues)

    #     evaluation = np.array([r[0] for r in regions])
    #     evalMinimum = np.min(evaluation)
    #     evalMaximum = np.max(evaluation)
    #     evaluation = (evaluation - evalMinimum) / \
    #         max(0.001, evalMaximum-evalMinimum)

    #     import matplotlib.patches as patches

    #     def plotInterest(region, ax, options):
    #         bounds = np.array(region[1])[cols]
    #         alpha = (1. + region[0]) / 2.
    #         if len(bounds) == 2:
    #             ax.add_patch(patches.Rectangle((bounds[0][0], bounds[1][0]), bounds[0][1] - bounds[0][0],
    #                                            bounds[1][1] - bounds[1][0], alpha=alpha, zorder=-1))
    #             ax.add_patch(patches.Rectangle((bounds[0][0], bounds[1][0]), bounds[0][1] - bounds[0][0],
    #                                            bounds[1][1] - bounds[1][0], alpha=alpha, zorder=-1, fill=False))
    #         else:
    #             ax.add_patch(patches.Rectangle(
    #                 (bounds[0][0], -1), bounds[0][1] - bounds[0][0], 2, alpha=alpha, zorder=-1))

    #     def lambdaInit(region):
    #         return lambda fig, ax, options: plotInterest(region, ax, options)

    #     bounds = np.array(self.finiteBounds())[cols]
    #     return getVisual(
    #         [lambdaInit(region) for region in regions] +
    #         [lambda fig, ax, options: plotData(
    #             points, fig, ax, options, color=pointValues, colorbar=True)],
    #         minimum=bounds[:, 0].tolist(),
    #         maximum=bounds[:, 1].tolist(),
    #         title='{}$Interest \in [{:.2f}, {:.2f}]$, $Progress \in [{:.2f}, {:.2f}]$\n{} points reached, {} regions'.format(prefix, evalMinimum,
    #                                                                                                                          evalMaximum, pvMinimum,
    #                                                                                                                          pvMaximum, points.shape[0], len(regions))
    #     )

    # Plot
    # def plot(self, outcomeOnly=True, contextOnly=False, absoluteProgress=False):
    #     visualize(self.getRegionsVisualizer(outcomeOnly=outcomeOnly,
    #                                         contextOnly=contextOnly, absoluteProgress=absoluteProgress))

    # *** Deprecated ***

    # def plot2(self, fig_id, color, norm):
    #     """
    #     Plot evaluation region as a rectangle patch which transparency indicates evaluation.

    #     fig_id int
    #     color string: indicates the colour of the patch
    #     norm float: used to normalize the patches transparency
    #     """
    #     num_plots = min(len(self.evaluation), 10)
    #     plt.ion()
    #     fig = plt.figure(fig_id)
    #     plt.show()
    #     if norm == 0.0:
    #         norm2 = 1.0
    #     else:
    #         norm2 = norm
    #     n = (num_plots + 1) / 2
    #     for s in range(num_plots):
    #         ax = fig.add_subplot(str(n) + "2" + str(s+1))
    #         if len(self.bounds['min']) == 1:
    #             p = patches.Rectangle(
    #                 (self.bounds['min'][0], -0.5), (self.bounds['max']
    #                                                 [0]-self.bounds['min'][0]), 1,
    #                 facecolor=color, alpha=-self.evaluation[s]/norm2)
    #             ax.add_patch(p)
    #         elif len(self.bounds['min']) == 2:
    #             p = patches.Rectangle(
    #                 (self.bounds['min'][0], self.bounds['min'][1]
    #                  ), (self.bounds['max'][0]-self.bounds['min'][0]),
    #                 (self.bounds['max'][1]-self.bounds['min'][1]),
    #                 facecolor=color, alpha=-self.evaluation[s]/norm2)
    #             ax.add_patch(p)
    #     plt.draw()

    # def plot_v2(self, norm, ax, options):
    #     """Plot evaluation region as a rectangle patch with transparency indicating evaluation."""
    #     num_plots = min(len(self.evaluation), 10)
    #     if norm == 0.0:
    #         norm2 = 1.0
    #     else:
    #         norm2 = norm
    #     n = (num_plots + 1) / 2
    #     if len(self.bounds['min']) == 1:
    #         p = patches.Rectangle(
    #             (self.bounds['min'][0], -0.5), (self.bounds['max']
    #                                             [0]-self.bounds['min'][0]), 1,
    #             facecolor=options['color'], alpha=-self.evaluation[s]/norm2)
    #         ax.add_patch(p)
    #     elif len(self.bounds['min']) == 2:
    #         p = patches.Rectangle(
    #             (self.bounds['min'][0], self.bounds['min'][1]
    #              ), (self.bounds['max'][0]-self.bounds['min'][0]),
    #             (self.bounds['max'][1]-self.bounds['min'][1]),
    #             facecolor=options['color'], alpha=-self.evaluation[s]/norm2)
    #         ax.add_patch(p)

    # Api
    # def apiget(self, range_=(-1, -1)):
    #     if self.parent:
    #         return {}
    #     # , 'evaluation': self.im.get_range(range_)}
    #     return {'regions': self.history.get_range(range_)}
