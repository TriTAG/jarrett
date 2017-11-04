import pyproj
import transitfeed
import json
from shapely.geometry import LineString,  Point
import os, sys
sys.path.insert(0,'/Users/mboos/OneDrive/dev/network-topology')
import network_topology as nt
from shapely.geometry import mapping
from functools import partial
import pickle
from graph import TransitGraph
from transitmap import SimpleMap

utm17 = pyproj.Proj(proj='utm', zone=17, ellps='WGS84')
google = pyproj.Proj(init='epsg:4326')
project = partial(pyproj.transform, utm17, google)

def getScheduleAndShapes():
    print('Loading GTFS...')
    loader = transitfeed.Loader('../GRT_GTFS_w17')
    schedule = loader.Load()

    linestrings = {}
    terminals = []


    print('Calculating {} shapes...'.format(len(schedule.GetShapeList())))
    for shape in schedule.GetShapeList():
        points = []
        for lat, lng, dst in shape.points:
            x, y = utm17(lng, lat)
            points.append([x, y])
        linestring = LineString(points)
        linestrings[shape.shape_id] = linestring
        for i in 0, -1:
            terminals.append(linestring.coords[i])

    terminals = [Point(p) for p in set(terminals)]

    print('Processing roundabouts...')
    roundabouts = []
    with open('Roads.geojson.json') as fp:
        roads = json.load(fp)
    for road in roads['features']:
        if road['properties']['CartoClass'] == 'Roundabout':
            points = zip(*utm17(*zip(*road['geometry']['coordinates'])))
            roundabouts.append(LineString(points))

    return schedule, linestrings, terminals, roundabouts


def getTopologyAndRoutes(routeLines, terminals, supplementaryLines, debugFolder=None):
    print('Calculating topology...')
    ls = list(routeLines.values()) + supplementaryLines
    G = nt.getNetworkTopology(ls, minInnerPerimeter=300,
                              splitAtTeriminals=terminals, turnThreshold=20.0)
    if debugFolder:
        with open(os.path.join(debugFolder, 'topology.pickle'), 'w') as fp:
            pickle.dump({'G': G, 'linestrings': routeLines}, fp)
        features = []
        for u, v, i, data in G.edges(data=True, keys=True):
            coord_list = [list(utm17(x, y, inverse=True))
                          for x, y in data['geom'].coords]
            features.append({"type": "Feature",
                             "properties": {'stroke-opacity': 0.25},
                             "geometry": {
                                "type": "LineString",
                                "coordinates": coord_list
                             }})
            if G[u][v][i]['terminal']:
                features[-1]['properties']['stroke'] = '#F00'
            else:
                features[-1]['properties']['stroke'] = '#00F'
        data = {"type": "FeatureCollection",
                "features": features}
        with open(os.path.join(debugFolder, 'topology.geojson'), 'w') as fp:
            json.dump(data, fp)

    selectedRoutes = routeLines
    # {s: l for i, (s, l) in enumerate(linestrings.items()) if i < 1000}
    print('Calculating paths...')
    paths = nt.getMatchedRoutes(selectedRoutes, G, increment=50, scope=20)

    if debugFolder:
        with open(os.path.join(debugFolder, 'paths.pickle'), 'w') as fp:
            pickle.dump(paths, fp)
        for shape_id, path in paths.items():
            ls = routeLines[shape_id]
            origShape = {"type": "Feature",
                         "properties": {'stroke-color': '#00F'},
                         "geometry": {
                            "type": "LineString",
                            "coordinates": [list(utm17(x, y, inverse=True))
                                            for x, y in ls.coords]
                         }}
            features = [origShape]
            for u, v, i in path:
                features.append({
                    "type": "Feature",
                    "properties": {'stroke': '#F00'},
                    "geometry": {
                        "type": "LineString",
                        'coordinates': [utm17(x, y, inverse=True)
                                        for x, y in G[u][v][i]['geom'].coords]
                    }
                })
            data = {"type": "FeatureCollection",
                    "features": features}
            # coords = []
            # for u, v, i in path:
            #     coords += list(G[u][v][i]['geom'].coords)
            # newline = LineString([utm17(x, y, inverse=True)
            #                       for x, y in coords])
            #
            # newShape = {"type": "Feature",
            #             "properties": {'stroke-color': '#F00'},
            #             "geometry": mapping(newline.simplify(1e-9))}
            # data = {"type": "FeatureCollection",
            #         "features": [origShape, newShape]}
            with open(os.path.join(debugFolder, '{}.geojson'.format(shape_id)),
                      'w') as fp:
                json.dump(data, fp)
    return G, paths


schedule, linestrings, terminals, roundabouts = getScheduleAndShapes()
# G, shapePaths = getTopologyAndRoutes(linestrings, terminals, roundabouts, 'debug')
#
# with open('jarret.pickle', 'w') as fp:
#     pickle.dump((G, shapePaths), fp)
with open('jarret.pickle') as fp:
    (G, shapePaths) = pickle.load(fp)

TG = TransitGraph(G, shapePaths, schedule)
M = SimpleMap(TG)
M.renderAsJson('test.geojson')
# import cProfile
# cProfile.run('runProcess()')
