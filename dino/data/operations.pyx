import numpy as np
from scipy.spatial.distance import euclidean


def getLid(lids, ids):
    a = lids[ids]
    return a[a >= 0]


def _nearestFromData(points, x, n=1, ignore=0):
    data = _computeDistances(points, x)
    i = data.argpartition(np.arange(ignore, min(n + ignore, data.shape[0])))
    i = i[ignore:n + ignore]
    return i, data[i]


def _nnFromData(lids, ids, data, n=1, ignore=0, restrictionIds=None, otherSpace=None):
    if otherSpace:
        if restrictionIds is not None:
            restrictionIds = list(set(restrictionIds).intersection(set(otherSpace.ids[:otherSpace.number])))
        else:
            restrictionIds = otherSpace.ids[:otherSpace.number]
    if restrictionIds is not None:
        try:
            lids = getLid(lids, restrictionIds)
            data = data[lids]
            ids = ids[lids]
        except Exception as e:
            raise e

    i = data.argpartition(np.arange(ignore, min(n + ignore, data.shape[0])))
    i = i[ignore:n + ignore]
    #i = data.argpartition(min(n, data.shape[0]))[:n]
    #i = i[data[i].argsort()]
    #i = i[ignore:min(i.size, n)]
    return ids[i], data[i]


def _computeDistances(data, x, weights=1., maxDist=1.):
    return (np.sum(((data - x) * weights) ** 2, axis=1) ** 0.5) / maxDist


def multivariateRegression(x, y, x0):
    """Compute a multivariate linear regression model y = f(x) using (x,y) and use it to compute f(x0)."""
    # code from scipy LinearRegression:
    x_offset = np.average(x, axis=0)
    y_offset = np.average(y, axis=0)

    coef_ = np.linalg.lstsq(x - x_offset, y - y_offset, rcond=None)[0]
    if y.ndim == 1:
        coef_ = np.ravel(coef_)

    intercept_ = y_offset - np.dot(x_offset, coef_)
    return np.array(x0).dot(coef_) + intercept_

    return np.array(x0).dot(np.linalg.lstsq(x, y, rcond=None)[0])

    # print("x")
    # print(x)
    # print("y")
    # print(y)
    # print("goal: {}".format(x0))

    x_ = np.ones((y.shape[0], x.shape[1] + 1))
    x_[:, 1:] = x

    theta = normalEquation(x_, y)
    #print(theta)

    #x_ = np.ones((1, x.shape[1] + 1))
    x_[0, 1:] = x0
    return x_[0, :].dot(theta)


def multivariateRegressionError(x, y, x0):
    # code from scipy LinearRegression:
    number = int(len(x) * 0.9)
    x_offset = np.average(x[:number], axis=0)
    y_offset = np.average(y[:number], axis=0)

    coef_ = np.linalg.lstsq(x[:number] - x_offset, y[:number] - y_offset, rcond=None)[0]
    if y.ndim == 1:
        coef_ = np.ravel(coef_)

    intercept_ = y_offset - np.dot(x_offset, coef_)
    y0 = np.array(x0).dot(coef_) + intercept_
    error = np.sum(np.square(x.dot(coef_) + intercept_ - y)) / np.sum(0.01 + np.square(y - np.mean(y)))
    # + 0.01 to avoid dividing by zero when all y are equal
    # error = 1. - r2_score(y, x.dot(coef_) + intercept_, multioutput='variance_weighted')

    return y0, error

    theta = np.linalg.lstsq(x, y, rcond=None)[0]
    std = np.mean(np.sqrt(np.sum(np.square(x.dot(theta) - y), axis=1)))
    return np.array(x0).dot(theta), std

    m = y.shape[0]
    #print(m)
    #print(x.shape)
    '''print("x")
    print(testSetX)
    print("y")
    print(testSetY)
    print("goal: {}".format(x0))'''
    x_ = np.ones((y.shape[0], x.shape[1] + 1))
    x_[:, 1:] = x

    theta = normalEquation(x_, y)
    #print(theta)
    #x_ = x_ if testSetX is None else np.hstack((np.ones((testSetY.shape[0], 1)), testSetX))
    #y = testSetY if testSetY is not None else y
    std = np.mean(np.sqrt(np.sum(np.square(x_.dot(theta) - y), axis=1)))
    #print(str(x1.dot(theta)[:, 0]) + "     " + str(y[:, 0]))
    #print(np.square(x1.dot(theta) - y))
    #print(np.sum(np.square(x1.dot(theta) - y), axis=1))
    #print(std)

    x_[0, 1:] = x0
    return x_[0, :].dot(theta), std


def normalEquation(x, y):
    """Compute the multivariate linear regression parameters using normal equation method."""
    #theta = np.zeros((x.shape[1], y.shape[1]))
    x2 = x.transpose()
    #xInv = np.linalg.pinv(x2.dot(x))
    #theta = xInv.dot(x2).dot(y)
    theta = np.linalg.pinv(x2.dot(x)).dot(x2).dot(y)
    return theta


# not used
def bestLocality(self, goal, goal_flat):
    """Compute most stable local action-outcome model around goal outcome."""
    # Compute best locality candidates
    ids, dist = self.outcomeSpace.nearest(goal, n=10, otherSpace=self.policySpace)

    min_points_y = 5
    min_points_a = 5

    # Check if distance to goal is too big and enough neighbours studied already
    ids_part = np.squeeze(np.argwhere(dist < self.outcomeSpace.options['dist']), axis=1)
    if len(ids_part) < min_points_y:
        ids_part = range(0, 5)
    ids = ids[ids_part]

    y_s = self.outcomeSpace.getPlainPoint(ids)
    a_s = self.policySpace.getPlainPoint(ids)
    #print(y_s)
    #print(a_s)

    best_y = np.array([])
    best_a = np.array([])
    min_distance = -1
    min_std = -1
    min_y0 = None
    min_a0 = None
    for i, a in enumerate(a_s):
        y = y_s[i]

        idsA, distA = self.policySpace.nearest(a, n=5, otherSpace=self.outcomeSpace)

        # Check if distance to a is too big and enough neighbours studied already
        ids_part = np.squeeze(np.argwhere(distA < self.policySpace.options['dist']), axis=1)
        if len(ids_part) < min_points_a:
            ids_part = range(0, 5)
        idsA = idsA[ids_part]

        y_all = self.outcomeSpace.getPlainPoint(idsA)
        a_all = self.policySpace.getPlainPoint(idsA)

        y_flat = np.array([x.flatten() for x in y_all])
        a_flat = np.array([x.flatten() for x in a_all])

        dd = 0.

        a0 = multivariateRegression(y_flat, a_flat, goal_flat)
        y0, std = multivariateRegressionError(a_flat, y_flat, a0)
        #y0, std = self.compute_outcome_from_action(a0, a_all, y_all)
        dd = euclidean(goal_flat, y0)
        #print(dd)

        # Create Multivariate Regression Model to compute mean model error
        '''for j in range(len(idsA)):
            # Compute regression model containing all but current action
            # Then compute action error for using model at y(current action)
            a_gen = multivariateRegression(y_flat[np.arange(len(y_flat)) != j, :], a_flat[np.arange(len(a_flat)) != j, :], y_flat[j, :])
            dd += euclidean(a_gen, a_flat[j, :]) / self.policySpace.options['max_dist']

        dd /= float(max(1, len(idsA)))'''

        if i == 0 or dd < min_distance:
            min_distance = dd
            min_std = std
            min_y0 = y0
            min_a0 = a0
            best_y = y_all
            best_a = a_all

    return best_y, best_a, min_y0, min_a0, min_std, min_distance
