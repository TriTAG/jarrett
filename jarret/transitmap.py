"""Blah blah."""
import math
import pyproj
import json
import numpy as np
from itertools import product, combinations, count
from shapely.geometry import Point, LineString
from util import firstNode, fullEdge, getDirection, intersection
from bisect import bisect_left
from scipy.interpolate import interp1d
from pulp import LpProblem, LpVariable, lpSum, LpStatus, value as lpvalue

class SimpleMap():
    def __init__(self, transitGraph, thickness=3, spacing=1, minRadius=10,
                 zoom=15, maxAngle=20.):
        self.graph = transitGraph.graph.copy()
        self.schedule = transitGraph.schedule
        bb = self.schedule.GetStopBoundingBox()
        latitude = (bb[0] + bb[2]) / 2.
        self.pixelSize = (156543.03392 * math.cos(latitude * math.pi / 180.) /
                          (2. ** zoom))
        self.thickness = thickness
        self.spacing = spacing
        self.totalThickness = (thickness + spacing) * self.pixelSize
        self.minRadius = minRadius
        self.maxAngle = np.deg2rad(maxAngle)

        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                self.graph.edges[edge]['sequence'] = []
                self.graph.edges[edge]['width'] = 0
                self.graph.edges[edge]['geom'] = self.graph.edges[edge]['geom'].simplify(self.totalThickness / 2.)
        self._sortRoutes()
        # for route in self.schedule.GetRouteList():
        #     if route.route_color != '':
        #         self._placeRoute(route.route_id)
        self._findOffsets()
        self._collapseEdges()

    def _sortRoutes(self):
        def edgeString(edge):
            (a, (a, b, i)), (b, (a, b, i)) = edge
            return '{}_{}_{}'.format(a, b, i)
        crossings = []
        seq_vars = {}
        pre_vars = {}

        problem = LpProblem('route_crossings')
        problem += 0, 'Minimize crossings'
        for edge in self.graph.edges:

            currentEdge = self.graph.edges[edge]
            edge = fullEdge(edge[0])
            if not currentEdge['junction']:
                routes = [r for r in sorted(currentEdge['routes'])
                          if self.schedule.GetRoute(r).route_color != '']
                N = len(routes)
                estr = 'edge_' + edgeString(edge) + '_{}'
                s = {r: LpVariable(estr.format(r), 0, N-1, cat='Integer')
                     for r in routes}
                seq_vars[edge] = s
                estr += '_{}'
                pre_vars[edge] = {}
                for (r1, r2) in combinations(routes, 2):
                    pre = LpVariable(estr.format(r1, r2), cat='Binary')
                    pre_vars[edge][r1, r2] = pre
                    problem += s[r1] + 1 <= s[r2] + N * (1 - pre), ''
                    problem += s[r2] + 1 <= s[r1] + N * pre, ''

        c_ctr = count()
        for edge in self.graph.edges:
            edge = fullEdge(edge[0])
            currentEdge = self.graph.edges[edge]
            if not currentEdge['junction']:
                routes = [r for r in currentEdge['routes']
                          if self.schedule.GetRoute(r).route_color != '']
                p_edge = pre_vars[edge]
                for end in edge:
                    currentNode = self.graph.node[end]
                    for other in currentNode['order']:
                        junction = self.graph.edges[end, other]
                        j_routes = [r for r in junction['routes'] if r in routes]
                        otherEdge = fullEdge(other)
                        p_other = pre_vars[otherEdge]
                        for r1, r2 in combinations(j_routes, 2):
                            if r2 < r1:
                                r1, r2 = r2, r1
                            if (r1, r2) not in p_edge or (r1, r2) not in p_other:
                                continue
                            c = LpVariable('c_{}'.format(c_ctr.next()), cat='Binary')
                            if firstNode(end) == firstNode(other):
                                problem += p_edge[r1, r2] + p_other[r1, r2] - 1 <= c, ''
                                problem += 1 - p_edge[r1, r2] - p_other[r1, r2] <= c, ''
                            else:
                                problem += p_edge[r1, r2] - p_other[r1, r2] <= c, ''
                                problem += p_other[r1, r2] - p_edge[r1, r2] <= c, ''
                            crossings.append(c)
                    for o1, o2 in combinations(currentNode['order'], 2):
                        j1 = self.graph.edges[end, o1]
                        j2 = self.graph.edges[end, o2]
                        jr1 = [r for r in j1['routes'] if r in routes]
                        jr2 = [r for r in j2['routes'] if r in routes]
                        for r1, r2 in product(jr1, jr2):
                            if r1 == r2:
                                continue
                            reverse = not firstNode(end)
                            if r1 > r2:
                                p = p_edge[r2, r1]
                                reverse = not reverse
                            else:
                                p = p_edge[r1, r2]
                            c = LpVariable('c_{}'.format(c_ctr.next()), cat='Binary')
                            if reverse:
                                problem += (1 - p == c), ''
                            else:
                                problem += p == c, ''
                            crossings.append(c*0.5)
        problem.setObjective(lpSum(crossings))
        #print problem
        problem.writeLP('crossings.lp')
        problem.solve()
        print lpvalue(problem.objective)

        # for c in crossings:
        #     print c.value()
        for edge, s_vars in seq_vars.items():
            currentEdge = self.graph.edges[edge]
            print edge
            for s, r in sorted([(s.value(), r) for (r, s) in s_vars.items()]):
                currentEdge['sequence'].append(r)
                print s, r
                currentEdge['width'] += self._getTotalThickness(edge, r)

    def _getTotalThickness(self, edge, route):
        if route in ['200', '201', '202', '203', '204', '7']:
            return (2 * self.thickness + self.spacing) * self.pixelSize
        return self.totalThickness

    def _getLineThickness(self, edge, route):
        if route in ['200', '201', '202', '203', '204', '7']:
            return self.thickness * 2
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
            skipped = []
            for edge in toKill:
                A, B = edge
                orderA = self.graph.node[A]['order']
                orderB = self.graph.node[B]['order']
                killingA = False
                killingB = False
                for O in orderA:
                    otherEdge = fullEdge(O)
                    if otherEdge in toKill:
                        killingA = True
                        break
                for O in orderB:
                    otherEdge = fullEdge(O)
                    if otherEdge in toKill:
                        killingB = True
                        break
                # if killingA and killingB:
                #     skipped.append(edge)
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
            for s in skipped:
                toKill.remove(s)

        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                geom = self.graph.edges[edge]['geom']
                starti, endi = 0, len(geom.coords)
                distances = [LineString([p, q]).length
                             for p, q in zip(geom.coords[:-1], geom.coords[1:])]
                distances = [0] + list(np.cumsum(distances))
                x, y = zip(*geom.coords)
                X = interp1d(distances, x)
                Y = interp1d(distances, y)
                for A in edge:
                    trim = self.graph.node[A]['trim']
                    if firstNode(A):
                        starti = bisect_left(distances, trim)
                        startP = [X(trim), Y(trim)]
                    else:
                        d = geom.length - trim
                        endi = bisect_left(distances, d)
                        endP = [X(d), Y(d)]
                coords = [startP] + geom.coords[starti:endi] + [endP]
                geom = LineString(coords).simplify(1)
                self.graph.edges[edge]['geom'] = geom
                # A, B = edge
                # A, B = fullEdge(A)
                # __, self.graph.node[A]['dir'] = getDirection(geom, True)
                # __, self.graph.node[B]['dir'] = getDirection(geom, False)


    def _calcEdgeTrim(self, A, currentEdge):
        T = np.array([[0, 1], [-1, 0]])
        trimA = 0
        endA, dA = getDirection(currentEdge['geom'], firstNode(A))
        # if firstNode(A):
        #     coordsA = np.array(currentEdge['geom'].coords[:2])
        # else:
        #     coordsA = np.array(currentEdge['geom'].coords[-1:-3:-1])
        # dA = np.diff(coordsA, axis=0)
        # dA /= np.linalg.norm(dA)
        # dA = np.squeeze(dA)
        self.graph.node[A]['dir'] = dA
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
                    pA = (endA + offsetA * np.dot(dA, T))

                    endO, dO = getDirection(otherEdge['geom'], firstNode(other))
                    if firstNode(other):
                    #     coordsO = np.array(otherEdge['geom'].coords[:2])
                        offsetO = otherEdge['offsets'][route]
                    else:
                    #     coordsO = np.array(otherEdge['geom'].coords[-1:-3:-1])
                        offsetO = -otherEdge['offsets'][route]
                    #
                    # dO = np.diff(coordsO, axis=0)
                    # dO /= np.linalg.norm(dO)
                    # dO = np.squeeze(dO)
                    pO = (endO + offsetO * np.dot(dO, T))
                    cosAO = np.dot(dA, dO)
                    sinAO = np.cross(dA, dO)
                    tAO2 = abs(sinAO) / max(1 + cosAO, 1e-300)  # trig ids!
                    setback = self.minRadius / tAO2
                    adjustment = (offsetA * cosAO - offsetO) / sinAO
                    trim = setback + adjustment + self.minRadius

                    #angle = np.arcsin(abs(sinAO))
                    self.graph.edges[A, other]['angle'] = np.arctan2(abs(sinAO), cosAO)
                    if np.sin(self.maxAngle) > np.sin(self.graph.edges[A, other]['angle']):
                    #if trim > 5 * self.minRadius:  # arbitrary
                        d = np.linalg.norm(pA - pO)
                        offsetA = max(currentEdge['offsets'].values()) - min(currentEdge['offsets'].values())
                        offsetO = max(otherEdge['offsets'].values()) - min(otherEdge['offsets'].values())
                        radius = self.minRadius + (offsetA + offsetO) / 2.
                        trim = np.sqrt(radius ** 2. -
                                       max(0, radius - d * .5) ** 2.) * 1.1
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
                currentEdge = self.graph.edges[edge]
                #self.graph.edges[edge]['offsets'] = {}
                self.graph.node[A]['points'] = {}
                self.graph.node[B]['points'] = {}
                centreGeom = currentEdge['geom'].simplify(self.totalThickness / 2.)
                __, self.graph.node[A]['dir'] = getDirection(centreGeom, True)
                __, self.graph.node[B]['dir'] = getDirection(centreGeom, False)
                #offset = -self.graph.edges[edge]['width'] / 2.
                for route in currentEdge['sequence']:
                    route_color = self.schedule.GetRoute(route).route_color
                    #offset += self._getTotalThickness(edge, route) / 2.
                    offset = currentEdge['offsets'][route]
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
                                            'stroke-opacity': 1,
                                            'route': route,
                                            'sides': str(self.graph.edges[edge]['sides'].items())
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
                seqA = list(edgeA['sequence'])
                seqB = list(edgeB['sequence'])
                if firstNode(A) == firstNode(B):
                    seqB.reverse()
                # __, dA = getDirection(edgeA['geom'], firstNode(A))
                # __, dB = getDirection(edgeB['geom'], firstNode(B))
                dA = self.graph.node[A]['dir']
                dB = self.graph.node[B]['dir']
                # if self.maxAngle > currentEdge['angle']:
                #
                #
                #     continue
                sinAB = np.cross(dA, dB)
                cosAB = np.dot(dA, dB)
                angle = np.arctan2(abs(sinAB), cosAB)
                right = sinAB > 0
                if right and firstNode(A) or not right and not firstNode(A):
                    seqA.reverse()
                    seqB.reverse()
                routesA = [r for r in seqA if r in currentEdge['routes']]
                routesB = [r for r in seqB if r in currentEdge['routes']]
                routes = [r for r in currentEdge['routes']
                          if r in routesA and r in routesB]
                if not routes:
                    continue
                offsetA0 = edgeA['offsets'][routesA[0]]
                offsetB0 = edgeB['offsets'][routesB[0]]

                # lastRadius = self.minRadius - self._getTotalThickness(edge, routeIndexes[0][1]) / 2.
                # delta = 0
                radii = {}
                for r in routes:
                    offsetA = abs(offsetA0 - edgeA['offsets'][r])
                    offsetB = abs(offsetB0 - edgeB['offsets'][r])
                    radii[r] = self.minRadius + min(offsetA, offsetB)
                # for i, r in sorted(routeIndexes):
                #     thickness = self._getTotalThickness(edge, r)
                #     if i != lastIndex:
                #         lastRadius += delta
                #         delta = 0
                #     radii[r] = lastRadius + thickness / 2.
                #     delta += thickness
                #     lastIndex = i
                for route in self.graph.edges[edge]['routes']:
                    route_color = self.schedule.GetRoute(route).route_color
                    if route not in edgeA['sequence'] or route not in edgeB['sequence']:
                        continue
                    ptA = np.array(self.graph.node[A]['points'][route])
                    ptB = np.array(self.graph.node[B]['points'][route])
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
                    radius = radii[route]
                    corner = intersection(ptA, dA, ptB, dB)
                    setback = radius / np.tan(angle/2)
                    delta = abs(setback-radius)/radius
                    filler = ''
                    if np.sin(self.maxAngle) > np.sin(angle) or delta > 1:
                        # hack for now with straight line
                        dAB = ptB - ptA
                        dAB /= np.linalg.norm(dAB)
                        dAR = np.array([dA[1], -dA[0]])
                        dBR = np.array([dB[1], -dB[0]])
                        crossAAB = np.cross(dA, dAB)
                        crossBAB = np.cross(dB, dAB)
                        dotAAB = np.dot(dA, dAB)
                        dotBAB = np.dot(dB, dAB)
                        offsetsA = [o
                                    for (r, o) in edgeA['offsets'].items()
                                    if r in routes]
                        offsetsB = [o
                                    for (r, o) in edgeB['offsets'].items()
                                    if r in routes]
                        if crossAAB < 0:  # left
                            if firstNode(A):
                                offsetA0 = min(offsetsA)
                            else:
                                offsetA0 = max(offsetsA)
                        else:  # right
                            if firstNode(A):
                                offsetA0 = max(offsetsA)
                            else:
                                offsetA0 = min(offsetsA)
                            dAR = -dAR
                        if crossBAB > 0:  # left
                            if firstNode(B):
                                offsetB0 = min(offsetsB)
                            else:
                                offsetB0 = max(offsetsB)
                        else:  # right
                            if firstNode(B):
                                offsetB0 = max(offsetsB)
                            else:
                                offsetB0 = min(offsetsB)
                            dBR = - dBR

                        radiusA = self.minRadius + abs(offsetA0 - edgeA['offsets'][route])
                        radiusB = self.minRadius + abs(offsetB0 - edgeB['offsets'][route])
                        centreA = ptA + dAR * radiusA
                        angleA = np.arctan2(-dAR[1], -dAR[0])
                        centreB = ptB + dBR * radiusB
                        angleB = np.arctan2(-dBR[1], -dBR[0])

                        l = np.linalg.norm(centreA - centreB)
                        #if radius + radius <= l:
                        __, dC = getDirection(LineString([centreA, centreB]))
                        angleC = np.arctan2(dC[1], dC[0])
                        if (crossAAB > 0 and crossBAB > 0) or (crossAAB < 0 and crossBAB < 0):
                            thetaB = np.arccos((radiusA-radiusB)/l)
                            thetaA = thetaB
                        elif radiusA + radiusB <= l:
                            thetaA = np.arccos((radiusA+radiusB)/l)
                            thetaB = np.pi - thetaA
                        else:
                            thetaA = 0
                            thetaB = np.pi
                        if crossAAB > 0:  # left
                            angleAC = angleC + thetaA
                        else:
                            angleAC = angleC - thetaA
                        if crossBAB < 0:  # left
                            angleBC = angleC - thetaB
                        else:
                            angleBC = angleC + thetaB
                        while angleAC - angleA > np.pi:
                            angleAC -= 2 * np.pi
                        while angleA - angleAC > np.pi:
                            angleAC += 2 * np.pi
                        while angleBC - angleB > np.pi:
                            angleBC -= 2 * np.pi
                        while angleB - angleBC > np.pi:
                            angleBC += 2 * np.pi
                        pts = [centreA + radiusA * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(angleA, angleAC, 5)]
                        pts += [centreB + radiusB * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(angleBC, angleB, 5)]
                        pts.insert(0, ptA)
                        pts.append(ptB)
                        filler = str((angleA, angleAC, angleBC, angleB, crossAAB, crossBAB, dotAAB, dotBAB, angleC, thetaA, thetaB, radiusA, radiusB))
                        # else:
                        #     pts = [ptA, ptB]

                        # coord_list = [list(utm17(*ptA, inverse=True)),
                        #               list(utm17(*ptB, inverse=True))]
                    else:
                        # radius = radii[route]
                        # corner = intersection(ptA, dA, ptB, dB)
                        # setback = radius / np.tan(angle/2)
                        endA = corner + dA * setback
                        if right:
                            # if firstNode(A):
                            #     setback = radius / np.tan(currentEdge['angle']/2)
                            # else:
                            #     setback = radius * np.tan(currentEdge['angle']/2)
                            # endA = corner + dA * setback
                            dAR = np.array([-dA[1], dA[0]])
                            dBR = np.array([dB[1], -dB[0]])
                            angleA = np.arctan2(-dAR[1], -dAR[0])
                            angleB = np.arctan2(-dBR[1], -dBR[0])
                            if angleB > angleA:
                                angleB -= 2. * np.pi
                        else:
                            # if firstNode(A):
                            #     setback = radius * np.tan(currentEdge['angle']/2)
                            # else:
                            #     setback = radius / np.tan(currentEdge['angle']/2)
                            # endA = corner + dA * setback
                            dAR = np.array([dA[1], -dA[0]])
                            dBR = np.array([-dB[1], dB[0]])
                            angleA = np.arctan2(-dAR[1], -dAR[0])
                            angleB = np.arctan2(-dBR[1], -dBR[0])
                            if angleB < angleA:
                                angleB += 2. * np.pi
                        centre = endA + radius * dAR
                        steps = int(abs(angleA - angleB)*4)  # for now, point at least every .25 radians
                        angles = np.linspace(angleA, angleB, steps + 2)
                        pts = [centre + [radius*np.cos(a), radius*np.sin(a)]
                               for a in angles]
                        pts.append(ptB)
                        pts.insert(0, ptA)
                    coord_list = [list(utm17(*pt, inverse=True))
                                  for pt in pts if pt[0] < 1e29]
                    thickness = min(self._getLineThickness(edgeA, route),
                                    self._getLineThickness(edgeB, route))
                    features.append({"type": "Feature",
                                     "properties": {
                                            'stroke-width': thickness,
                                            'stroke': '#' + route_color,
                                            'stroke-opacity': 1,
                                            'angles': filler
                                            # 'setback': min(setback, 1e30),
                                            # 'radius': radii[route]
                                        },
                                     "geometry": {
                                        "type": "LineString",
                                        "coordinates": coord_list
                                     }})


        data = {"type": "FeatureCollection",
                "features": features}
        with open(filename, 'w') as fp:
            json.dump(data, fp)
