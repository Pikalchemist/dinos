import cython
import numpy as np
cimport cython
cimport numpy as np
from libc.math cimport pow, sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def getLid(lids, ids):
    if ids is None:
        return None
    a = lids[ids]
    return a[a >= 0]


@cython.boundscheck(False)
@cython.wraparound(False)
def nearestNeighborsFromData(points, x, n=1, ignore=0):
    data = euclidean_distances_numpy(points, x)
    i = data.argpartition(np.arange(ignore, min(n + ignore, data.shape[0])))
    if n > 0:
        i = i[ignore:n + ignore]
    return i, data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
def nearestNeighborsFromDataContiguous(points, x, n=1, ignore=0):
    data = euclidean_distances(points, x)
    i = data.argpartition(np.arange(ignore, min(n + ignore, data.shape[0])))
    if n > 0:
        i = i[ignore:n + ignore]
    return i, data[i]


@cython.boundscheck(False)
@cython.wraparound(False)
def findRestrictionsLids(restrictionIds, space, otherSpace):
    if otherSpace and not (otherSpace.contiguous and otherSpace.ids[otherSpace._number - 1] >= space.ids[space._number - 1]):
        if restrictionIds is not None:
            restrictionIds = np.intersect1d(restrictionIds, otherSpace.ids[:otherSpace._number])
        else:
            restrictionIds = otherSpace.ids[:otherSpace._number]
    return getLid(space.lids, restrictionIds)


@cython.boundscheck(False)
@cython.wraparound(False)
def nearestNeighbors(ids, data, n=1, ignore=0, restrictionLids=None, otherSpace=None):
    if restrictionLids is not None:
        ids = ids[restrictionLids]

    i = data.argpartition(np.arange(ignore + 1, min(n + ignore, data.shape[0])))
    i = i[ignore:n + ignore]
    return ids[i], data[i]

#, np.int32_t[::1] lids=None
def euclidean_distances(double[:, ::1] data, double[::1] goal, double[::1] weights=None, double maxDist=1., np.uint8_t[::1] columns=None):
    if weights is None:
        return np.asarray(_euclidean_distances(data, goal, maxDist, columns))
    else:
        return np.asarray(_euclidean_distances_weighted(data, goal, weights, maxDist, columns))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double[::1] _euclidean_distances(double[:, ::1] vectors_a, double[::1] vectors_b, double maxDist, np.uint8_t[::1] columns):
    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_dims = vectors_a.shape[1]
    cdef double[::1] distances = np.empty([numb_vectors_a])

    cdef int i, k
    cdef double distance, temp

    if columns is None:
        for i in range(numb_vectors_a):
            distance = 0.0
            for k in range(numb_dims):
                temp = vectors_a[i, k] - vectors_b[k]
                distance += (temp*temp)

            distances[i] = sqrt(distance) / maxDist
    else:
        for i in range(numb_vectors_a):
            distance = 0.0
            for k in range(numb_dims):
                if columns[k]:
                    temp = vectors_a[i, k] - vectors_b[k]
                    distance += (temp*temp)

            distances[i] = sqrt(distance) / maxDist

    return distances


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef double[::1] _euclidean_distances_weighted(double[:, ::1] vectors_a, double[::1] vectors_b, double[::1] weights, double maxDist, np.uint8_t[::1] columns):
    cdef int numb_vectors_a = vectors_a.shape[0]
    cdef int numb_dims = vectors_a.shape[1]
    cdef double[::1] distances = np.empty([numb_vectors_a])

    cdef int i, k
    cdef double distance, temp

    if columns is None:
        for i in range(numb_vectors_a):
            distance = 0.0
            for k in range(numb_dims):
                temp = vectors_a[i, k] - vectors_b[k]
                distance += (temp*temp) * weights[k]

            distances[i] = sqrt(distance) / maxDist
    else:
        for i in range(numb_vectors_a):
            distance = 0.0
            for k in range(numb_dims):
                if columns[k]:
                    temp = vectors_a[i, k] - vectors_b[k]
                    distance += (temp*temp) * weights[k]

            distances[i] = sqrt(distance) / maxDist

    return distances


@cython.boundscheck(False)
@cython.wraparound(False)
def euclidean_distances_numpy(data, x, weights=1., maxDist=1.):
    return (np.sum(((data - x) * weights) ** 2, axis=1) ** 0.5) / maxDist


@cython.boundscheck(False)
@cython.wraparound(False)
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

    # return np.array(x0).dot(np.linalg.lstsq(x, y, rcond=None)[0])


@cython.boundscheck(False)
@cython.wraparound(False)
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

    # theta = np.linalg.lstsq(x, y, rcond=None)[0]
    # std = np.mean(np.sqrt(np.sum(np.square(x.dot(theta) - y), axis=1)))
    # return np.array(x0).dot(theta), std


def normalEquation(x, y):
    """Compute the multivariate linear regression parameters using normal equation method."""
    #theta = np.zeros((x.shape[1], y.shape[1]))
    x2 = x.transpose()
    #xInv = np.linalg.pinv(x2.dot(x))
    #theta = xInv.dot(x2).dot(y)
    theta = np.linalg.pinv(x2.dot(x)).dot(x2).dot(y)
    return theta

