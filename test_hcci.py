import json
import numpy as np
import random
import topologika
import topologika_reference
import unittest


class TestHCCI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.fromfile('../data/hcci_rr_oh_0001_560x560x560_float32.raw', dtype=np.float32).reshape(560, 560, 560)
        cls.forest = topologika.MergeForest(cls.data)
        cls.forest_reference = topologika_reference.MergeForest(cls.data)


    def test_maxima_query(self):
        maxima = self.forest.query_maxima()
        maxima_set = set(maxima)
        self.assertEqual(len(maxima), len(maxima_set))

        maxima_reference = self.forest_reference.query_maxima()
        maxima_reference_set = set(maxima_reference)
        self.assertEqual(len(maxima_reference), len(maxima_reference_set))

        self.assertEqual(maxima_set, maxima_reference_set)


    def test_component_query(self):
        random.seed(0)
        for _ in range(50):
            vertex = random.randrange(0, self.data.shape[0]*self.data.shape[1]*self.data.shape[2])
            threshold = random.uniform(0, 1)

            component = self.forest.query_component(vertex, threshold)
            component_reference = self.forest_reference.query_component(vertex, threshold)

            if not component:
                self.assertEqual(component, component_reference)
                continue

            component_set = set(component)
            self.assertEqual(len(component), len(component_set))

            component_reference_set = set(component_reference)
            self.assertEqual(len(component_reference), len(component_reference_set))

            self.assertEqual(component_set, component_reference_set)
