"""Blah blah."""
import math
import pyproj
import json
import numpy as np
from itertools import product
from shapely.geometry import Point, LineString
from util import firstNode, fullEdge, getDirection
from bisect import bisect_left
from scipy.interpolate import interp1d

class SimpleMap():
    def __init__(self, transitGraph, thickness=3, spacing=1, minRadius=10,
                 zoom=15):
        self.graph = transitGraph.graph.copy()
        self.schedule = transitGraph.schedule
        bb = self.schedule.GetStopBoundingBox()
        latitude = (bb[0] + bb[2]) / 2.
        self.pixelSize = (156543.03392 * math.cos(latitude * math.pi / 180.) /
                          (2. ** zoom))
        self.thickness = thickness
        self.totalThickness = (thickness + spacing) * self.pixelSize
        self.minRadius = minRadius

        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                self.graph.edges[edge]['sequence'] = []
                self.graph.edges[edge]['width'] = 0
        for route in self.schedule.GetRouteList():
            if route.route_color != '':
                self._placeRoute(route.route_id)
        self._findOffsets()
        self._collapseEdges()

    def _placeRoute(self, route_id):
        startEdge = None
        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                if route_id in self.graph.edges[edge]['routes']:
                    startEdge = edge
                    break
        if not startEdge:
            return
        visited = []
        A, B = startEdge
        startEdge = fullEdge(A)
        stack = [(startEdge, None, None, None)]
        while stack:
            nextEdge, nextNode, lastEdge, lastNode = stack.pop()
            if nextEdge in visited:
                continue
            visited.append(nextEdge)
            ll = 1e30
            lr = -1e30
            A, B = nextEdge
            self.graph.edges[nextEdge]['width'] += self._getTotalThickness(nextEdge, route_id)
            currentOrder = self.graph.edges[nextEdge]['sequence']
            if nextNode is not None:
                lastOrder = self.graph.edges[lastEdge]['sequence']
                # if firstNode(nextNode) != firstNode(lastNode):
                #     lastOrder = list(reversed(lastOrder))
                if set(lastOrder) == set(currentOrder + [route_id]):
                    if firstNode(nextNode) != firstNode(lastNode):
                        lr = lastOrder.index(route_id)
                    else:
                        lr = len(currentOrder) - lastOrder.index(route_id)
                    ll = lr + 1
                else:
                    found = False
                    for other_route in lastOrder:
                        if other_route == route_id:
                            found = True
                        elif not found and other_route in currentOrder:
                            if self.graph.edges[lastEdge]['sides'][route_id, other_route] == 1:
                                ll = currentOrder.index(other_route)
                        elif found and other_route in currentOrder:
                            if self.graph.edges[lastEdge]['sides'][route_id, other_route] == -1:
                                lr = currentOrder.index(other_route)
                                break
            for i in range(max(lr, 0), min(ll, len(currentOrder))):
                if route_id != currentOrder[i] and self.graph.edges[nextEdge]['sides'][route_id, currentOrder[i]] == 1:
                    currentOrder.insert(i, route_id)
                    break
            if route_id not in currentOrder:
                currentOrder.append(route_id)

            a, e = A
            for N in self.graph.node[A]['order']:
                edge = fullEdge(N)
                if route_id in self.graph.edges[edge]['routes']:
                    stack.append((edge, N, nextEdge, A))
            b, e = B
            for N in self.graph.node[B]['order']:
                edge = fullEdge(N)
                if route_id in self.graph.edges[edge]['routes']:
                    stack.append((edge, N, nextEdge, B))

    def _getTotalThickness(self, edge, route):
        return self.totalThickness

    def _getLineThickness(self, edge, route):
        return self.thickness

    def _findOffsets(self):
        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                offset = -self.graph.edges[edge]['width'] / 2.
                self.graph.edges[edge]['offsets'] = {}
                for route in self.graph.edges[edge]['sequence']:
                    thickness = self._getTotalThickness(edge, route)
                    offset += thickness / 2.
                    self.graph.edges[edge]['offsets'][route] = offset
                    offset += thickness / 2.

    def _collapseEdges(self):
        toKill = True
        while toKill:
            toKill = []
            for edge in self.graph.edges:
                currentEdge = self.graph.edges[edge]
                if not currentEdge['junction']:
                    totalTrim = 0
                    for A in edge:
                        trim = self._calcEdgeTrim(A, currentEdge)
                        self.graph.node[A]['trim'] = trim
                        totalTrim += trim
                    if totalTrim > currentEdge['geom'].length:
                        toKill.append(edge)
            print 'edges to remove: ', len(toKill)
            for edge in toKill:
                A, B = edge
                print 'removing', edge
                orderA = self.graph.node[A]['order']
                orderB = self.graph.node[B]['order']
                for endA, endB in product(orderA, orderB):
                    routesA = self.graph.edges[A, endA]['routes']
                    routesB = self.graph.edges[B, endB]['routes']
                    routes = {r: list(set(routesA[r]).intersection(routesB[r]))
                              for r in set(routesA).intersection(set(routesB))}
                    if (endA, endB) not in self.graph.edges:
                        self.graph.add_edge(endA, endB, junction=True, routes=routes)
                for endA in orderA:
                    order = self.graph.node[endA]['order']
                    i = order.index(A)
                    self.graph.node[endA]['order'] = order[0:i] + orderB + order[i+1:]
                for endB in orderB:
                    order = self.graph.node[endB]['order']
                    i = order.index(B)
                    self.graph.node[endB]['order'] = order[0:i] + orderA + order[i+1:]
                self.graph.remove_nodes_from(edge)

        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                geom = self.graph.edges[edge]['geom']
                starti, endi = 0, len(geom.coords)
                distances = [LineString([p, q]).length
                             for p, q in zip(geom.coords[:-1], geom.coords[1:])]
                distances = [0] + list(np.cumsum(distances))
                print edge
                print distances
                x, y = zip(*geom.coords)
                X = interp1d(distances, x)
                Y = interp1d(distances, y)
                for A in edge:
                    trim = self.graph.node[A]['trim']
                    if firstNode(A):
                        print 'A', trim
                        starti = bisect_left(distances, trim)
                        startP = [X(trim), Y(trim)]
                    else:
                        print 'B', trim
                        d = geom.length - trim
                        endi = bisect_left(distances, d)
                        endP = [X(d), Y(d)]
                coords = [startP] + geom.coords[starti:endi] + [endP]
                self.graph.edges[edge]['geom'] = LineString(coords)

    def _calcEdgeTrim(self, A, currentEdge):
        T = np.array([[0, 1], [-1, 0]])
        trimA = 0
        if firstNode(A):
            coordsA = np.array(currentEdge['geom'].coords[:2])
        else:
            coordsA = np.array(currentEdge['geom'].coords[-1:-3:-1])
        dA = np.diff(coordsA, axis=0)
        dA /= np.linalg.norm(dA)
        dA = np.squeeze(dA)
        for other in self.graph.node[A]['order']:
            if (A, other) in self.graph.edges:
                for route in self.graph.edges[A, other]['routes']:
                    if route not in currentEdge['offsets']:
                        continue
                    otherEdge = self.graph.edges[fullEdge(other)]
                    if firstNode(A):
                        offsetA = currentEdge['offsets'][route]
                    else:
                        offsetA = -currentEdge['offsets'][route]
                    pA = (coordsA[0] + offsetA * np.dot(dA, T))
                    if firstNode(other):
                        coordsO = np.array(otherEdge['geom'].coords[:2])
                        offsetO = otherEdge['offsets'][route]
                    else:
                        coordsO = np.array(otherEdge['geom'].coords[-1:-3:-1])
                        offsetO = -otherEdge['offsets'][route]

                    dO = np.diff(coordsO, axis=0)
                    dO /= np.linalg.norm(dO)
                    dO = np.squeeze(dO)
                    pO = (coordsO[0] + offsetO * np.dot(dO, T))
                    cosAO = np.dot(dA, dO)
                    sinAO = np.cross(dA, dO)
                    tAO2 = abs(sinAO) / max(1 + cosAO, 1e-300)  # trig ids!
                    setback = self.minRadius / tAO2
                    adjustment = (offsetA * cosAO - offsetO) / sinAO
                    trim = setback + adjustment
                    print A, setback, adjustment
                    if np.sin(np.deg2rad(20.)) > abs(sinAO):
                    #if trim > 5 * self.minRadius:  # arbitrary
                        d = np.linalg.norm(pA - pO)
                        trim = np.sqrt(self.minRadius ** 2. -
                                       max(0, self.minRadius - d * .5) ** 2.)
                        print 'trim', trim, d
                    trimA = max(trimA, trim)
        return trimA

    def renderAsJson(self, filename):
        bb = self.schedule.GetStopBoundingBox()
        longitude = (bb[1] + bb[3]) / 2.
        zone = (int((longitude + 180)/6) % 60) + 1
        utm17 = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
        features = []

        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                A, B = edge
                A, B = fullEdge(A)
                #self.graph.edges[edge]['offsets'] = {}
                self.graph.node[A]['points'] = {}
                self.graph.node[B]['points'] = {}
                centreGeom = self.graph.edges[edge]['geom'].simplify(self.totalThickness / 2.)
                #offset = -self.graph.edges[edge]['width'] / 2.
                for route in self.graph.edges[edge]['sequence']:
                    route_color = self.schedule.GetRoute(route).route_color
                    #offset += self._getTotalThickness(edge, route) / 2.
                    offset = self.graph.edges[edge]['offsets'][route]
                    geom = centreGeom.parallel_offset(offset, 'left')
                    #self.graph.edges[edge]['offsets'][route] = offset
                    if offset >= 0:
                        self.graph.node[A]['points'][route] = geom.coords[0]
                        self.graph.node[B]['points'][route] = geom.coords[-1]
                    else:
                        self.graph.node[A]['points'][route] = geom.coords[-1]
                        self.graph.node[B]['points'][route] = geom.coords[0]
                    #offset += self._getTotalThickness(edge, route) / 2.
                    coord_list = [list(utm17(inverse=True, *c))
                                  for c in geom.coords]
                    features.append({"type": "Feature",
                                     "properties": {
                                            'stroke-width': self._getLineThickness(edge, route),
                                            'stroke': '#' + route_color,
                                            'sequence': list(self.graph.edges[edge]['sequence']),
                                            # 'sides': str(self.graph.edges[edge]['sides'].items())
                                        },
                                     "geometry": {
                                        "type": "LineString",
                                        "coordinates": coord_list
                                     }})
        for edge in self.graph.edges:
            if self.graph.edges[edge]['junction']:
                A, B = edge
                currentEdge = self.graph.edges[edge]
                edgeA = self.graph.edges[fullEdge(A)]
                edgeB = self.graph.edges[fullEdge(B)]
                seqA = edgeA['sequence']
                seqB = edgeB['sequence']
                # if not firstNode(A):
                #     seqB.reverse()
                # if firstNode(B):
                #     seqA.reverse()
                # if isNegCrossProd:
                #     seqA.reverse()
                #     seqB.reverse()
                routesA = [r for r in seqA if r in currentEdge['routes']]
                routesB = [r for r in seqB if r in currentEdge['routes']]
                for route in self.graph.edges[edge]['routes']:
                    route_color = self.schedule.GetRoute(route).route_color
                    if route not in edgeA['sequence'] or route not in edgeB['sequence']:
                        continue
                    ptA = self.graph.node[A]['points'][route]
                    ptB = self.graph.node[B]['points'][route]
                    # offsetA = edgeA['offsets'][route]
                    # offsetB = edgeB['offsets'][route]
                    # if firstNode(A):
                    #     ptA = edgeA['geom'].interpolate(0, normalized=True)
                    #     dirA = getDirection(edgeA['geom'], 0)
                    # else:
                    #     ptA = edgeA['geom'].interpolate(1, normalized=True)
                    #     dirA = getDirection(edgeA['geom'], 1)
                    # if firstNode(B):
                    #     ptB = edgeB['geom'].interpolate(0, normalized=True)
                    #     dirB = getDirection(edgeB['geom'], 0)
                    # else:
                    #     ptB = edgeB['geom'].interpolate(1, normalized=True)
                    #     dirB = getDirection(edgeB['geom'], 1)
                    # ptA = Point(ptA.x - dirA[1] * offsetA, ptA.y + dirA[0] * offsetA)
                    # ptB = Point(ptB.x - dirB[1] * offsetB, ptB.y + dirB[0] * offsetB)
                    coord_list = [list(utm17(*ptA, inverse=True)),
                                  list(utm17(*ptB, inverse=True))]
                    # hack for now with straight line
                    features.append({"type": "Feature",
                                     "properties": {
                                            'stroke-width': self.thickness,
                                            'stroke': '#' + route_color,
                                        },
                                     "geometry": {
                                        "type": "LineString",
                                        "coordinates": coord_list
                                     }})


        data = {"type": "FeatureCollection",
                "features": features}
        with open(filename, 'w') as fp:
            json.dump(data, fp)
