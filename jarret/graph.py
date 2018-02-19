import networkx as nx
import numpy as np
import math
from itertools import combinations, cycle
from util import edgeKey, firstNode, fullEdge, getDirection
from transitfeed import Trip


class TransitGraph():
    """Representation of transit routes to help facilitate mapping."""

    def __init__(self, graph, paths, schedule):
        """Construct graph from topology, matched routes, and schedule."""
        self.graph = nx.Graph()
        self.schedule = schedule
        print 'build graph'
        self._buildGraph(graph)
        print 'map trips to graph'
        self._mapTripsToGraph(paths)
        # print 'compute order'
        # self._computeOrder()

    def _buildGraph(self, graph):
        for u in graph:
            connections = []
            directions = []
            for v in graph[u]:
                for i in graph[u][v]:
                    a, b, i = edgeKey(u, v, i)
                    A, B = (a, (a, b, i)), (b, (a, b, i))
                    geom = graph[a][b][i]['geom'].simplify(1e-3)
                    if (A, B) not in self.graph.edges:
                        self.graph.add_edge(A, B, geom=geom,
                                            junction=False, routes={},
                                            sides={})
                    connections.append((a, b, i))
                    __, d = getDirection(geom, firstNode((u, (a, b, i))))
                    #print A, B, d
                    directions.append(math.atan2(d[1], d[0]))
                    self.graph.node[(u, (a, b, i))]['order'] = []
            for A, B in combinations(connections, 2):
                self.graph.add_edge((u, A), (u, B), junction=True, routes={})

            curA = None
            numDirs = 0
            for d, A in cycle(sorted(zip(directions, connections))):
                if curA is None:
                    if numDirs == len(connections):
                        break
                    curA = A
                    numDirs += 1
                elif A == curA:
                    curA = None
                else:
                    self.graph.node[(u, curA)]['order'].append((u, A))
                # Code review from Junia, 15 months:
                # msvv      x      Qqgg      ygA

    def _mapTripsToGraph(self, paths):
        """Determine which routes run along which segments."""
        for trip in self.schedule.GetTripList():
            route = trip.route_id
            lastNode = None
            for (u, v, i) in paths[trip.shape_id]:
                a, b, i = edgeKey(u, v, i)
                A, B = (a, (a, b, i)), (b, (a, b, i))
                if route not in self.graph.edges[A, B]['routes']:
                    self.graph.edges[A, B]['routes'][route] = []
                trips = self.graph.edges[A, B]['routes'][route]
                if lastNode and lastNode[1] == (a, b, i):
                    trips[-1] = (trip.trip_id, 0)
                else:
                    if u == a:
                        trips.append((trip.trip_id, 1))
                    else:
                        trips.append((trip.trip_id, -1))
                    if lastNode:
                        c, C = lastNode
                        if route not in self.graph.edges[(c, C), (u, (a, b, i))]['routes']:
                            self.graph.edges[(c, C), (u, (a, b, i))]['routes'][route] = []
                        trips = self.graph.edges[(c, C), (u, (a, b, i))]['routes'][route]
                        if C < (a, b, i):
                            trips.append((trip.trip_id, 1))
                        else:
                            trips.append((trip.trip_id, -1))
                lastNode = (v, (a, b, i))
        self._linkBlocks(paths)

    def _linkBlocks(self, paths):
        blocks = {}
        for trip in self.schedule.GetTripList():
            bid = trip.block_id
            if bid is None:
                continue
            if bid not in blocks:
                blocks[bid] = []
            blocks[bid].append(trip)
        for bid, trips in blocks.items():
            trips = sorted(trips, key=Trip.GetStartTime)
            for t1, t2 in zip(trips[:-1], trips[1:]):
                p1 = paths[t1.shape_id]
                p2 = paths[t2.shape_id]
                u, v, i = p1[-1]
                x, y, j = p2[0]
                if (t1.route_id == t2.route_id and v == x and
                   (u, v, i) != (x, y, j) and u != y):
                    A = edgeKey(u, v, i)
                    B = edgeKey(x, y, j)
                    if t1.route_id not in self.graph.edges[(v, A), (x, B)]['routes']:
                        self.graph.edges[(v, A), (x, B)]['routes'][t1.route_id] = []
                    trips = self.graph.edges[(v, A), (x, B)]['routes'][t1.route_id]
                    if A < B:
                        trips.append((t1.trip_id, 1))
                    else:
                        trips.append((t2.trip_id, -1))



    def _computeOrder(self):
        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                routes = self.graph.edges[edge]['routes']
                for route1, route2 in combinations(routes, 2):
                    if not (route1, route2) in self.graph.edges[edge]['sides']:
                        self._compareRoutes(edge, route1, route2)

    def _compareRoutes(self, startEdge, route1, route2):
        total = 0
        visited = {}
        startEdge = fullEdge(startEdge[0])
        queue = [(startEdge, 1)]
        while queue:
            (A, B), direction = queue.pop()
            if (A, B) in visited:
                continue
            visited[A, B] = direction
            total -= direction * self._getSideVotes(A, route1, route2)
            total += direction * self._getSideVotes(B, route1, route2)

            a, (u, v, i) = A
            for nextNode in self.graph.nodes[A]['order']:
                nextEdge = fullEdge(nextNode)
                if (route1 in self.graph.edges[nextEdge]['routes'] and
                        route2 in self.graph.edges[nextEdge]['routes']):
                    if firstNode(nextNode):
                        queue.insert(0, (nextEdge, -direction))
                    else:
                        queue.insert(0, (nextEdge, direction))

            b, (u, v, i) = B
            for nextNode in self.graph.nodes[B]['order']:
                nextEdge = fullEdge(nextNode)
                if (route1 in self.graph.edges[nextEdge]['routes'] and
                        route2 in self.graph.edges[nextEdge]['routes']):
                    if firstNode(nextNode):
                        queue.insert(0, (nextEdge, direction))
                    else:
                        queue.insert(0, (nextEdge, -direction))

        if total >= 0:  # break ties in favour of right
            side = 1
        else:
            side = -1
        for edge, direction in visited.items():
            self.graph.edges[edge]['sides'][route1, route2] = side * direction
            self.graph.edges[edge]['sides'][route2, route1] = -side * direction

    def _getSideVotes(self, node, route1, route2):
        """Determine orientation of route 1 to 2 at junction.

        positive if 1 right of 2 looking out from an end node
        negative if 1 left of 2 looking out from an end node
        negative if 1 right of 2 looking out from a start node
        positive if 1 left of 2 looking out from a start node
        """
        n, (u, v, i) = node
        tot = 0
        found1 = []
        found2 = []
        for i, turn in enumerate(self.graph.node[node]['order']):
            routes = self.graph.edges[node, turn]['routes']
            if route1 in routes:
                found1.append(i)
            if route2 in routes:
                found2.append(i)

        for a in found1:
            for b in found2:
                if a < b:
                    tot += 1
                elif b < a:
                    tot -= 1

        return tot
