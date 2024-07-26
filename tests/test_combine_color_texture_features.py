import unittest
import numpy as np
from src.color_feature import *
from src.texture_feature import *
from src.combine_color_texture_features import combine_color_texture_features

class TestCombineColorTextureFeaturesExtraction(unittest.TestCase):
    def testCombineFeatures(self):
        FILENAME = '../tests/test_images/platypus.jpg'
        combinedFeatures = combine_color_texture_features(FILENAME)

        self.assertIsInstance(combinedFeatures, np.ndarray)

        expected_length = 512 + 256 #should be 80?
        self.assertEqual(len(combinedFeatures), expected_length)

if __name__ == '__main__':
    unittest.main()