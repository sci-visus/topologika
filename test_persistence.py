import math
import numpy as np
import topologika
import unittest


class TestPersistence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # TODO: topologika reads the following 3D numpy array in an incorrect way (we cast directly
        #   to a pointer but that produces incorrect result)
        #cls.data = np.array([[
        #    [10, 5, 11],
        #    [0, 6, 1],
		#]], dtype=np.float32)
        cls.data = np.array([10, 5, 11, 0, 6, 1], dtype=np.float32).reshape(1, 2, 3)
        cls.forest = topologika.MergeForest(cls.data)


    def test_maxima_query(self):
        maxima = self.forest.query_maxima()
        array = self.data.flatten()
        self.assertEqual(set([array[m] for m in maxima]), set([10, 11]))


    def test_component_max_query(self):
        self.assertEqual(self.forest.query_component_max(2, 5), 2)
        self.assertEqual(self.forest.query_component_max(0, 5), 2)


    def test_persistence_query(self):
        persistence = self.forest.query_persistence(0)
        self.assertEqual(persistence, 5)

        persistence = self.forest.query_persistence(2)
        self.assertEqual(persistence, math.inf)