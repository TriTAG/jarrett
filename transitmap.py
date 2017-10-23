"""Blah blah."""
import math
import pyproj
import json

class SimpleMap():
    def __init__(self, transitGraph, thickness=2, spacing=1, zoom=15):
        self.graph = transitGraph.graph.copy()
        self.schedule = transitGraph.schedule
        bb = self.schedule.GetStopBoundingBox()
        latitude = (bb[0] + bb[2]) / 2.
        self.pixelSize = (156543.03392 * math.cos(latitude * math.pi / 180.) /
                          (2. ** zoom))
        self.thickness = thickness
        self.totalThickness = (thickness + spacing) * self.pixelSize

        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                self.graph.edges[edge]['sequence'] = []
                self.graph.edges[edge]['width'] = 0
        for route in self.schedule.GetRouteList():
            if route.route_color != '':
                self._placeRoute(route.route_id)

    def _placeRoute(self, route_id):
        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                if route_id in self.graph.edges[edge]['routes']:
                    self.graph.edges[edge]['width'] += self.totalThickness
                    if len(self.graph.edges[edge]['sequence']) == 0:
                        self.graph.edges[edge]['sequence'] = [route_id]
                    else:
                        # ll = 1e30
                        # lr = le-30
                        found = False
                        for i, other in enumerate(self.graph.edges[edge]['sequence']):
                            if self.graph.edges[edge]['sides'][other, route_id] < 0:
                                self.graph.edges[edge]['sequence'].insert(i, route_id)
                                found = True
                                break
                        if not found:
                            self.graph.edges[edge]['sequence'].append(route_id)

    def renderAsJson(self, filename):
        bb = self.schedule.GetStopBoundingBox()
        longitude = (bb[1] + bb[3]) / 2.
        zone = (int((longitude + 180)/6) % 60) + 1
        utm17 = pyproj.Proj(proj='utm', zone=zone, ellps='WGS84')
        features = []

        for edge in self.graph.edges:
            if not self.graph.edges[edge]['junction']:
                centreGeom = self.graph.edges[edge]['geom'].simplify(self.totalThickness / 2.)
                offset = -self.graph.edges[edge]['width'] / 2.
                for route in self.graph.edges[edge]['sequence']:
                    route_color = self.schedule.GetRoute(route).route_color
                    offset += self.totalThickness / 2.
                    geom = centreGeom.parallel_offset(offset, 'left')
                    offset += self.totalThickness / 2.
                    coord_list = [list(utm17(inverse=True, *c))
                                  for c in geom.coords]
                    features.append({"type": "Feature",
                                     "properties": {
                                            'stroke-width': self.thickness,
                                            'stroke': '#' + route_color,
                                            'routes': list(self.graph.edges[edge]['routes'])
                                        },
                                     "geometry": {
                                        "type": "LineString",
                                        "coordinates": coord_list
                                     }})

        data = {"type": "FeatureCollection",
                "features": features}
        with open(filename, 'w') as fp:
            json.dump(data, fp)
