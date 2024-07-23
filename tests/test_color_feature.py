import unittest
import numpy as np
import cv2
from color_feature import extract_color_features

class TestColorFeatureExtraction(unittest.TestCase):
    def test_extract_color_features(self):
        # Create a test image (solid red color)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:] = [0, 0, 255]  # Red in BGR

        # Save the test image to a temporary file
        test_image_filename = 'test_image.jpg'
        cv2.imwrite(test_image_filename, test_image)

        # Extract color features
        color_features = extract_color_features(test_image_filename)

        # Check that the feature vector is of the correct length
        self.assertEqual(len(color_features), 72)

        # Check that the sum of the normalized feature vector is 1 (since it's normalized)
        self.assertAlmostEqual(np.sum(color_features), 1.0)

        # Clean up the test image file
        import os
        os.remove(test_image_filename)

if __name__ == '__main__':
    unittest.main() 
    