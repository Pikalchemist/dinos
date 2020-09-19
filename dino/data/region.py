import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import copy
import random
import pickle

from exlab.utils.text import strtab
from exlab.interface.graph import Graph

from .space import Space, SpaceKind
from .dataspace import DataSpace
# from ..utils.logging import DataEventHistory, DataEventKind
# from ..utils.io import strtab, getVisual, plotData, visualize
# from dino.utils.maths import uniformSampling, first
from .data import Data


class SpaceRegion(object):
    """Implements an evaluation region."""

    def __init__(self, targetSpace, options, bounds=None, parent=None, manager=None, tag='', contextSpace=None,
                 regions=None):
        """
        bounds float list dict: contains min and max boundaries of the region
        options dict: different options used by the evaluation model
        """
        self.space = targetSpace.spaceManager.multiColSpace(
            [targetSpace, contextSpace], weight=0.5)
        self.targetSpace = targetSpace
        self.contextSpace = contextSpace
        self.parent = parent
        self.manager = manager
        self.bounds = copy.deepcopy(
            bounds) if bounds else Space.infiniteBounds(self.space.dim)
        assert(parent is not None or manager is not None)
        self.tag = tag

        self.colsContext = self.space.columnsFor(self.contextSpace)
        self.colsTarget = self.space.columnsFor(self.targetSpace)

        # Contains the following keys:
        #   'minSurface': the minimum surface of a region to be still splittable
        #   'maxPoints': the maximum of points contained in a region (used to split)
        #   'window': the number of last progress measures to consider for computing region evaluation (to take newer
        #             points only)
        #   'cost': cost of strategy (usually strategies involving a teacher have a higher cost)
        self.options = {
            "cost": 1.0,
            "window": 25,
            "maxAttempts": 20,
            "maxPoints": 50,
            "minSurface": 0.05
        }
        self.options.update(options)

        self.points = []
        self.pointValues = []
        self.evaluation = 0.
        '''for i in range(len(options['costs'])):
            self.points.append([])
            self.pointValues.append([])
            self.evaluation.append(0.0)'''

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

    def __repr__(self):
        return "Region " + str(self.space) + "\n    Left: " + strtab(self.leftChild) + "\n    Cut " + strtab(str(self.splitValue) + " / " + str(self.splitDim) + " #" + str(len(self.points[0]))) + "\n    Right: " + strtab(self.rightChild)

    def _serialize(self, options):
        # serialize(self, ['name', 'description', 'effector', 'property', 'options'])
        dict_ = {}
        dict_['bounds'] = self.bounds
        dict_['id'] = self.lid
        return dict_

    def root(self):
        root = self
        while root.parent is not None:
            root = root.parent
        return root

    def finiteBounds(self):
        points = np.array(self.root().getSplitData(withRegions=False)[0])
        return [(np.min(points[:, j]) if bmin == -math.inf else bmin,
                 np.max(points[:, j]) if bmax == math.inf else bmax)
                for j, (bmin, bmax) in enumerate(self.bounds)]

    def maxDistance(self):
        _bounds = list(zip(np.min(self.points, axis=0).tolist(),
                           np.max(self.points, axis=0).tolist()))
        return (sum([(bound[1] - bound[0]) ** 2 for bound in _bounds])) ** 0.5

    @property
    def splitten(self):
        return self.leftChild is not None

    def nearestContext(self, context):
        self.contextSpace._validate()
        context = context.convertTo(kind=SpaceKind.PRE).projection(
            self.contextSpace).plain()

        indices, distances = DataSpace.nearestFromData(np.array(self.points)[:, self.colsContext], context,
                                                       n=self.options['maxPoints']//2)

        return indices, distances

    def nearContext(self, context, tolerance=0.01):
        if not self.contextSpace:
            return True

        self.contextSpace._validate()
        context = context.convertTo(kind=SpaceKind.PRE).projection(
            self.contextSpace).plain()
        tolerance *= self.contextSpace.maxDistance

        # Context in bounds
        inBounds = True
        for i, col in enumerate(self.colsContext):
            if context[i] < self.bounds[col][0] - tolerance or context[i] > self.bounds[col][1] + tolerance:
                inBounds = False
                break
        return inBounds

    def controllableContext(self, dataset):
        return dataset.controllableSpaces(self.contextSpace)

    def addPoint(self, point, progress, firstAlwaysNull=True):
        """Add a point and its progress in the attached evaluation region."""
        point = Data.plainData(point, self.space)
        assert len(point) == len(self.bounds)

        # print("ADDING 1 POINT", point)
        # if filterNull and np.all(np.array(point)[self.colsTarget]==0):
        #     print("NULL POINT")
        #     return

        if firstAlwaysNull and len(self.points) == 0:
            progress = 0.
        self.points.append(point)
        self.pointValues.append(progress)
        self.computeEvaluation()
        addedInChildren = False

        maxPoints = self.options['maxPoints']
        if not self.splitten and self.splittable and len(self.points) > maxPoints:
            # Leave reached
            # Region must be splat
            self.randomCut(self.options['maxAttempts'])
            #self.greedyCut()
            # Making sure a cut has been found
            if self.splitDim is not None:
                self.split()
        elif self.splitten:
            # Traverse tree
            addedInChildren = True
            if point[self.splitDim] < self.splitValue:
                self.leftChild.addPoint(point, progress)
            else:
                self.rightChild.addPoint(point, progress)
        # Remove oldest points and pointValues
        if len(self.points) > maxPoints:
            self.points = self.points[-maxPoints:]
            self.pointValues = self.pointValues[-maxPoints:]

        if not addedInChildren and self.manager:
            self.manager.logger.debug('Adding point {} with progress {} to region {}'.format(
                point, progress, self), self.tag)

    def computeEvaluation(self):
        pass

    # Splitting process
    def setSplittable(self):  # Change it to use max depth tree instead ???
        """Check if the region is splittable according to its surface."""
        surface = 1.
        for bmin, bmax in self.bounds:
            surface *= (bmax - bmin)
        self.splittable = (surface > self.options['minSurface'])

    def split(self):
        """Split region according to the cut decided beforehand."""

        # Create child regions boundaries
        leftBounds = copy.deepcopy(self.bounds)
        leftBounds[self.splitDim][1] = self.splitValue
        rightBounds = copy.deepcopy(self.bounds)
        rightBounds[self.splitDim][0] = self.splitValue

        # Create empty child regions
        self.leftChild = self.__class__(self.targetSpace, self.options, bounds=leftBounds,
                                        parent=self, contextSpace=self.contextSpace, regions=self.regions)
        self.rightChild = self.__class__(self.targetSpace, self.options, bounds=rightBounds,
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
        for point, progress in zip(self.points, self.pointValues):
            if point[self.splitDim] < self.splitValue:
                self.leftChild.addPoint(point, progress)
            else:
                self.rightChild.addPoint(point, progress)

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
            i.sort(key=lambda j: self.points[j][d])
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

                splitValue = (self.points[i[idcut-1]]
                              [d] + self.points[i[idcut]][d]) / 2.

                # Make sure we are not trying to split something unsplittable
                if self.points[i[idcut-1]][d] == splitValue:
                    continue

                left = []
                right = []
                for j in range(len(self.points)):
                    if self.points[j][d] < splitValue:
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
                    progress_left += self.pointValues[j]
                progress_left /= float(max(len(left_filtered), 1))
                for j in right_filtered:
                    progress_right += self.pointValues[j]
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

    def getSplitData(self, withRegions=True):
        points = list(self.points)
        pointValues = list(self.pointValues)

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
        g = Graph(title='Regions from {}'.format(self), options=options)
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
            cols = self.colsTarget
            points = points[:, cols]
        else:
            cols = np.arange(self.space.dim)

        # pvMinimum = np.min(pointValues)
        # pvMaximum = np.max(pointValues)
        evaluation = np.array([r[0] for r in regions])
        evalMinimum = np.min(evaluation)
        # evalMaximum = np.max(evaluation)
        evaluation = (evaluation - evalMinimum) / \
            max(0.001, np.max(evaluation) - evalMinimum)

        for region in regions:
            bounds = np.array(region[1])[cols]
            alpha = (1. + region[0]) / 2.
            if len(bounds) == 2:
                g.rectangle((bounds[0][0], bounds[1][0]), bounds[0][1] - bounds[0]
                            [0], bounds[1][1] - bounds[1][0], alpha=alpha, zorder=-1)
            else:
                g.rectangle((bounds[0][0], -1), bounds[0][1] -
                            bounds[0][0], 2, alpha=alpha, zorder=-1)
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
