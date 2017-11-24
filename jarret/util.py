import numpy as np
from shapely.geometry import LineString


def edgeKey(u, v, i):
    """Return source, target pair in sorted order."""
    if u > v:
        return v, u, i
    else:
        return u, v, i


def firstNode(A):
    u, (a, b, i) = A
    return u == a


def otherNode(A):
    u, (a, b, i) = A
    if firstNode(A):
        return b, (a, b, i)
    else:
        return a, (a, b, i)


def fullEdge(A):
    if firstNode(A):
        return A, otherNode(A)
    else:
        return otherNode(A), A


def getDirection(line, fromStart=True):
    """Unit vector in direction of line at start or end."""
    if fromStart:
        coordsA = np.array(line.coords[:2])
    else:
        coordsA = np.array(line.coords[-1:-3:-1])
    dA = np.diff(coordsA, axis=0)
    norm = np.linalg.norm(dA)
    # if norm < 1e-12:
    #     if fromStart:
    #         return getDirection(LineString(line.coords[1:]), True)
    #     else:
    #         return getDirection(LineString(line.coords[:-1]), False)
    dA /= norm
    dA = np.squeeze(dA)
    return coordsA[0], dA

    # if distance == line.length:
    #     n1 = 1. - 1e-6
    #     n2 = 1.
    # else:
    #     n1 = distance / line.length
    #     n2 = n1 + 1e-6
    # p1 = line.interpolate(n1, normalized=True)
    # p2 = line.interpolate(n2, normalized=True)
    # v = np.array([p2.x - p1.x, p2.y - p1.y])
    # return v / np.linalg.norm(v)


def intersection(pA, dA, pB, dB):
    t = (dB[1]*(pB[0]-pA[0])-dB[0]*(pB[1]-pA[1]))/(dA[0]*dB[1]-dB[0]*dA[1])
    return np.array([pA[0]+dA[0]*t, pA[1]+dA[1]*t])
