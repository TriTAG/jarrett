import unittest
from ..graph import TransitGraph
from ..transitmap import SimpleMap
import pyproj
import networkx as nx
import transitfeed
from shapely.geometry import LineString
import random

rawCoords = {
    70: [-80.48768520355225, 43.436623180360456],
    71: [-80.4828143119812, 43.43501829815333],
    80: [-80.48313617706299, 43.43871102166051],
    81: [-80.45167922973633, 43.45578498947261],
    82: [-80.45167922973633, 43.450426525819836],
    83: [-80.4627513885498, 43.454258498771445],
    84: [-80.4887580871582, 43.44101691191008],
    85: [-80.48536777496338, 43.4357506283238],
    88: [-80.44163703918457, 43.46793323547617],
    89: [-80.43580055236816, 43.45796562363117],
    90: [-80.47626972198485, 43.44394588902309],
    91: [-80.45687198638915, 43.452638507312024],
    92: [-80.44343948364258, 43.46220205946993],
    93: [-80.45643210411072, 43.449265982046676],
    94: [-80.4572582244873, 43.45746719989944],
    100: [-80.46712875366211, 43.45017728339725],
    101: [-80.44803142547607, 43.46740374743117],
    102: [-80.43631553649902, 43.464133277299865]
}

edgeNodes = [(70, 80), (70, 84), (70, 85),
             (71, 80), (71, 85),
             (80, 84), (80, 85), (80, 90),
             (81, 91), (81, 92),
             (82, 91), (82, 93),
             (83, 91), (83, 94),
             (88, 92),
             (89, 92),
             (90, 100),
             (91, 93), (91, 94), (91, 100),
             (92, 101), (92, 102),
             (93, 100)]


class SideTestCase(unittest.TestCase):

    def test_simple1(self):
        """Test side sorting for routes for simple, clear-cut case: ][ ."""
        TG = TransitGraph(self.G, self.paths, self.schedule)
        expected = [
            (80, 90, 0, 1), (90, 100, 0, 1), (91, 100, 0, -1), (81, 91, 0, -1),
            (81, 92, 0, 1)
        ]
        for a, b, i, side in expected:
            edge = TG.graph.edges[(a, (a, b, i)), (b, (a, b, i))]
            self.assertEqual(edge['sides']['shape1', 'shape2'], side,
                             'Segment ({}, {}, {}) has wrong order for routes'.format(a, b, i))
            self.assertEqual(edge['sides']['shape2', 'shape1'], -side,
                             'Segment ({}, {}, {}) has wrong reverse order for routes'.format(a, b, i))

    def test_P(self):
        """Test side sorting for routes for simple case with loop: |P ."""
        TG = TransitGraph(self.G, self.paths, self.schedule)
        expected = [
            (80, 90, 0, 1), (90, 100, 0, 1), (91, 100, 0, -1), (81, 91, 0, -1),
            (81, 92, 0, 1), (80, 85, 0, -1)
        ]
        for a, b, i, side in expected:
            edge = TG.graph.edges[(a, (a, b, i)), (b, (a, b, i))]
            self.assertEqual(edge['sides']['shape1', 'shape5'], side,
                             'Segment ({}, {}, {}) has wrong order for routes'.format(a, b, i))
            self.assertEqual(edge['sides']['shape5', 'shape1'], -side,
                             'Segment ({}, {}, {}) has wrong reverse order for routes'.format(a, b, i))

    def test_crosses(self):
        """Test side sorting for routes for simple, clear-cut case: ][ ."""
        TG = TransitGraph(self.G, self.paths, self.schedule)
        expected = [
            (80, 90, 0, -1), (90, 100, 0, -1), (91, 100, 0, 1), (81, 91, 0, 1),
            (81, 92, 0, -1)
        ]
        for a, b, i, side in expected:
            edge = TG.graph.edges[(a, (a, b, i)), (b, (a, b, i))]
            self.assertEqual(edge['sides']['shape3', 'shape4'], side,
                             'Segment ({}, {}, {}) has wrong order for routes'.format(a, b, i))
            self.assertEqual(edge['sides']['shape4', 'shape3'], -side,
                             'Segment ({}, {}, {}) has wrong reverse order for routes'.format(a, b, i))

    def test_Pcrosses(self):
        """Test side sorting for routes for loop and crossing: LP ."""
        TG = TransitGraph(self.G, self.paths, self.schedule)
        expected = [
            (80, 90, 0, 1), (90, 100, 0, 1), (91, 100, 0, -1), (81, 91, 0, -1),
            (81, 92, 0, 1), (80, 85, 0, -1)
        ]
        for a, b, i, side in expected:
            edge = TG.graph.edges[(a, (a, b, i)), (b, (a, b, i))]
            self.assertEqual(edge['sides']['shape5', 'shape4'], side,
                             'Segment ({}, {}, {}) has wrong order for routes'.format(a, b, i))
            self.assertEqual(edge['sides']['shape4', 'shape5'], -side,
                             'Segment ({}, {}, {}) has wrong reverse order for routes'.format(a, b, i))

    def test_Render(self):
        TG = TransitGraph(self.G, self.paths, self.schedule)
        M = SimpleMap(TG)
        M.renderAsJson('test2.geojson')
        xxx

    def setUp(self):
        utm17 = pyproj.Proj(proj='utm', zone=17, ellps='WGS84')

        lc = {k: utm17(*v) for k, v in rawCoords.items()}
        G = nx.MultiDiGraph()
        for e1, e2 in edgeNodes:
            G.add_edge(e1, e2, geom=LineString([lc[e1], lc[e2]]))
            G.add_edge(e2, e1, geom=LineString([lc[e2], lc[e1]]))
        self.G = G

        schedule = transitfeed.Schedule()
        schedule.AddAgency("Mike's bus", "http://google.com",
                           "America/Toronto")
        stop1 = schedule.AddStop(lng=-80.46, lat=43.45, name="Dummy stop 1")
        stop2 = schedule.AddStop(lng=-80.47, lat=43.45, name="Dummy stop 2")

        self.paths = {
            'shape1': [(85, 80, 0), (80, 90, 0), (90, 100, 0), (100, 91, 0),
                       (91, 81, 0), (81, 92, 0), (92, 89, 0)],
            'shape2': [(84, 80, 0), (80, 90, 0), (90, 100, 0), (100, 91, 0),
                       (91, 81, 0), (81, 92, 0), (92, 101, 0)],
            'shape3': [(84, 80, 0), (80, 90, 0), (90, 100, 0), (100, 91, 0),
                       (91, 81, 0), (81, 92, 0), (92, 89, 0)],
            'shape4': [(85, 80, 0), (80, 90, 0), (90, 100, 0), (100, 91, 0),
                       (91, 81, 0), (81, 92, 0), (92, 101, 0)],
            'shape5': [(102, 92, 0), (92, 81, 0), (81, 91, 0), (91, 100, 0),
                       (100, 90, 0), (90, 80, 0), (80, 85, 0), (85, 70, 0),
                       (70, 80, 0)],
            'shape6': [(94, 91, 0), (91, 100, 0), (100, 90, 0), (90, 80, 0),
                       (80, 71, 0), (71, 85, 0), (85, 80, 0)]
        }
        colours = {
            'shape1': 'ff0000',
            'shape2': '00ff00',
            'shape3': '0000ff',
            'shape4': 'ff00ff',
            'shape5': '669900',
            'shape6': '996600'
        }

        for p in self.paths:
            route = schedule.AddRoute(short_name="{}".format(p),
                                      long_name="Route {}".format(p),
                                      route_type="Bus",
                                      route_id='{}'.format(p))
            route.route_color = colours[p] #"{:06x}".format(random.randint(0, 0xFFFFFF))
            schedule.AddShapeObject(transitfeed.Shape(p))
            trip = route.AddTrip(schedule, headsign="To Downtown")
            trip.AddStopTime(stop1, stop_time='09:00:00')
            trip.AddStopTime(stop2, stop_time='09:15:00')
            trip.shape_id = p

        self.schedule = schedule


if __name__ == '__main__':
    unittest.main()