import json
import numpy as np
import random
import topologika
import topologika_reference
import unittest


class TestRandom(unittest.TestCase):
    def setUp(self):
        self.data = np.random.rand(128, 128, 128).astype(np.float32) # assumes region size 64^3
        self.forest = topologika.MergeForest(self.data)
        self.forest_reference = topologika_reference.MergeForest(self.data)


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