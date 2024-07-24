import unittest
import numpy as np
from color_feature import *
from texture_feature import *

class TestCombineColorTextureFeaturesExtraction(unittest.TestCase):
    def testCombineFeatures(self):
        FILENAME = '../tests/test_images/platypus.jpg'
        combinedFeatures = combine_color_texture_features(FILENAME)

        self.assertIsInstance(combined_features, np.ndarray)

        expected_length = 512 + 256
        self.assertEqual(len(combined_features), expected_length)

if __name__ == '__main__':
    unittest.main()