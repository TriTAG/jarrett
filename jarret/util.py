import numpy as np


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

def getDirection(line, distance):
    """Unit vector in direction of line at normalized distance."""
    if distance == line.length:
        n1 = 1. - 1e-6
        n2 = 1.
    else:
        n1 = distance / line.length
        n2 = n1 + 1e-6
    p1 = line.interpolate(n1, normalized=True)
    p2 = line.interpolate(n2, normalized=True)
    v = np.array([p2.x - p1.x, p2.y - p1.y])
    return v / np.linalg.norm(v)
