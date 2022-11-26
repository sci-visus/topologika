import math
import numpy as np
import random
import topologika as ta
import unittest



class TestAPI(unittest.TestCase):
    def test_construction(self):
        forest = ta.MergeForest(np.random.rand(128, 128, 128).astype(np.float32))
        forest = ta.MergeForest(np.random.rand(128, 128, 128).astype(np.float32), [32, 32, 32])
        with self.assertRaises(ValueError):
            forest = ta.MergeForest(np.random.rand(128, 128, 128).astype(np.float32), [32, 32])
        with self.assertRaises(ValueError):
            forest = ta.MergeForest(np.random.rand(128, 128, 128).astype(np.float32), [32, 32, 32, 32])
        # TODO: region size > domain size should work correctly; TEST!
        forest = ta.MergeForest(np.random.rand(128, 128, 128).astype(np.float32), [256, 256, 256])
        with self.assertRaises(TypeError):
            forest = ta.MergeForest(np.random.rand(128, 128, 128).astype(np.float32), [32.1, 32, 32])
        with self.assertRaises(TypeError):
            forest = ta.MergeForest(np.random.rand(128, 128, 128).astype(np.float32), ['hello', 32, 32])
        with self.assertRaises(ValueError):
            forest = ta.MergeForest(np.random.rand(128, 128, 128).astype(np.float32), [-1, -2, -3])
        with self.assertRaises(ValueError):
            forest = ta.MergeForest(np.random.rand(128, 128, 128).astype(np.uint8))
        forest = ta.MergeForest(array=np.random.rand(128, 128, 128).astype(np.float32))
        forest = ta.MergeForest(array=np.random.rand(128, 128, 128).astype(np.float32), region_shape=[32, 32, 32])

    def test_maxima_query(self):
        forest = ta.MergeForest(np.random.rand(128, 128, 128).astype(np.float32))
        ta.maxima(forest)

    def test_componentmax_query(self):
        data = np.random.rand(128, 128, 128).astype(np.float32)
        forest = ta.MergeForest(data)
        ta.componentmax(forest, (10, 0, 0), 0.1)
        ta.componentmax(forest, vertex=(10, 0, 0), threshold=0.1)
        with self.assertRaises(ValueError):
            ta.componentmax(forest, (-1, 0, 0), 0.1)
        with self.assertRaises(ValueError):
            ta.componentmax(forest, (data.shape[0], 0, 0), 0.1)
        with self.assertRaises(TypeError):
            ta.componentmax()
        with self.assertRaises(TypeError):
            ta.componentmax(forest, vertex=(10, 0, 0))
        with self.assertRaises(TypeError):
            ta.componentmax(forest, vertex=(10, 0, 0))
        self.assertIsNone(ta.componentmax(forest, (0, 0, 0), float('inf')))

    def test_component_query(self):
        data = np.random.rand(128, 128, 128).astype(np.float32)
        forest = ta.MergeForest(data)
        ta.component(forest, (10, 0, 0), 0.1)
        ta.component(forest, vertex=(10, 0, 0), threshold=0.1)
        with self.assertRaises(ValueError):
            ta.component(forest, (-1, 0, 0), 0.1)
        with self.assertRaises(ValueError):
            ta.component(forest, (data.shape[0], 0, 0), 0.1)
        with self.assertRaises(TypeError):
            ta.component()
        with self.assertRaises(TypeError):
            ta.component(forest, vertex=(10, 0, 0))
        # TODO: should we return an empty component instead of None?
        self.assertIsNone(ta.component(forest, (0, 0, 0), float('inf')))

    def test_components_query(self):
        data = np.random.rand(128, 128, 128).astype(np.float32)
        forest = ta.MergeForest(data)
        ta.components(forest, 0.1)
        ta.components(forest, threshold=0.1)
        self.assertEqual(ta.components(forest, float('inf')), [])
        with self.assertRaises(TypeError):
            ta.components()



class TestPersistenceAndTriplet(unittest.TestCase):
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
            [[25, 18, 20],
             [24, 10, 19],
             [22, 9, 23]],
            [[0, 1, 2],
             [4, 5, 6],
             [3, 8, 7]],
            [[13, 14, 15],
             [12, 26, 16],
             [11, 21, 17]]], dtype=np.float32)
        cls.forest = ta.MergeForest(cls.data, region_shape=[2, 2, 2])

    def test_maxima_query(self):
        maxima = ta.maxima(self.forest)
        self.assertEqual(set(maxima), set([(0, 0, 0), (0, 0, 2), (0, 2, 2), (2, 1, 1)]))

    def test_persistence_query(self):
        self.assertEqual(ta.persistence(self.forest, (0, 0, 0)), 17)
        self.assertEqual(ta.persistence(self.forest, (0, 0, 2)), 1)
        self.assertEqual(ta.persistence(self.forest, (0, 2, 2)), 5)
        self.assertEqual(ta.persistence(self.forest, (2, 1, 1)), math.inf)
        self.assertEqual(ta.persistence(self.forest, (1, 0, 0)), 0)

    def test_triplet_query(self):
        self.assertEqual(ta.triplet(self.forest, (0, 0, 0)), ((0, 0, 0), (1, 2, 1), (2, 1, 1)))
        self.assertEqual(ta.triplet(self.forest, (0, 0, 2)), ((0, 0, 2), (0, 1, 2), (0, 2, 2)))
        self.assertEqual(ta.triplet(self.forest, (0, 2, 2)), ((0, 2, 2), (0, 0, 1), (0, 0, 0)))
        self.assertEqual(ta.triplet(self.forest, (2, 1, 1)), None)
