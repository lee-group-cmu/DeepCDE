import numpy as np
import unittest2 as unittest

from utils import box_transform, normalize


class MyTest(unittest.TestCase):

    def test__box_transform(self):
        example_array = np.arange(0, 1, 0.01).round(2)

        self.assertRaises(ValueError, box_transform, example_array, 0.1, 1)
        self.assertRaises(ValueError, box_transform, example_array, 0, 0.8)
        self.assertRaises(ValueError, box_transform, example_array, 0.1, 0.8)

        self.assertTrue((box_transform(np.arange(0, 10, 0.1), 0, 10).round(2) == example_array).all())
        self.assertTrue((box_transform(np.arange(0, 100, 1), 0, 100).round(2) == example_array).all())
        self.assertTrue((box_transform(np.arange(-50, 50, 1), -50, 50).round(2) == example_array).all())


    def test__normalize(self):
        example_range = np.arange(0, 1, 0.01)
        example_density = np.random.normal(loc=0, scale=0.1, size=example_range.shape)
        idx_negative = (example_density <= 0)
        epsilon = 0.05

        # Normalization in place
        normalize(example_density)

        self.assertTrue((1- epsilon) < np.trapz(example_density, example_range) < (1 + epsilon))
        self.assertTrue(np.sum(example_density[idx_negative]) == 0)


if __name__ == '__main__':
    unittest.main()