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
