import math
import numpy as np
import topologika
import unittest




class TestTriplet1D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.array([4, 3, 1, 0, 2, 5], dtype=np.float32).reshape(1, 1, 6)
        cls.forest = topologika.MergeForest(cls.data, region_dims=[2, 2, 2])

    def test_maxima_query(self):
        self.assertEqual(set(self.forest.query_maxima()), set([0, 5]))

    def test_component_max_query(self):
        self.assertEqual(self.forest.query_component_max(0, 4), 0)
        self.assertEqual(self.forest.query_component_max(0, 0), 5)

    def test_triplet_query(self):
        self.assertEqual(self.forest.query_triplet(0), (0, 3, 5))
        self.assertEqual(self.forest.query_triplet(5), None)



class TestTriplet2D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # TODO: topologika reads the following 3D numpy array in an incorrect way (we cast directly
        #   to a pointer but that produces incorrect result)
        #cls.data = np.array([[
        #    [10, 5, 11],
        #    [0, 6, 1],
        #]], dtype=np.float32)
        cls.data = np.array([10, 5, 11, 0, 6, 1], dtype=np.float32).reshape(1, 2, 3)
        cls.forest = topologika.MergeForest(cls.data, region_dims=[2, 2, 2])

    def test_maxima_query(self):
        self.assertEqual(set(self.forest.query_maxima()), set([0, 2]))

    def test_component_max_query(self):
        self.assertEqual(self.forest.query_component_max(2, 5), 2)
        self.assertEqual(self.forest.query_component_max(0, 5), 2)

    def test_triplet_query(self):
        self.assertEqual(self.forest.query_triplet(0), (0, 1, 2))
        self.assertEqual(self.forest.query_triplet(2), None)




class TestTriplet3D(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # superlevel-set merge tree as index (value) nodes
        #                                22 (26)
        # 0 (25)                        /
        #    \                8 (23) /
        #     \    2 (20)   /     /
        #      \       \      /     /
        #       \      5 (19)   /
        #        \       /        /
        #         1 (18)      /
        #             \         /
        #              16 (8)
        #                 |
        cls.data = np.array([
            25, 18, 20,
            24, 10, 19,
            22, 9, 23,
            0, 1, 2,
            4, 5, 6,
            3, 8, 7,
            13, 14, 15,
            12, 26, 16,
            11, 21, 17], dtype=np.float32).reshape(3, 3, 3)
        cls.forest = topologika.MergeForest(cls.data, region_dims=[2, 2, 2])

    def test_maxima_query(self):
        maxima = self.forest.query_maxima()
        self.assertEqual(set(maxima), set([0, 2, 8, 22]))

    def test_triplet_query(self):
        self.assertEqual(self.forest.query_triplet(0), (0, 16, 22))
        self.assertEqual(self.forest.query_triplet(2), (2, 5, 8))
        self.assertEqual(self.forest.query_triplet(8), (8, 1, 0))
        self.assertEqual(self.forest.query_triplet(22), None)
