import unittest
import numpy as np 
import cv2 
from color_feature import extract_color_features

class TestColorFeatureExtraction(unittest. TestCase):
    def test_extract_color_features(self):

        #create a test image(solid red color)
        test_image = np.zeros((100, 100, 3), dtype=np.unit8)
        test_image[:] = [0, 0, 25]

        #save test image to temporary file
        test_image_filename = 'test_image.jpg'
        cv2.imwrite(test_image_filename, test_image)

        #extract color features
        color_features = extract_color_features(test_image_filename)

        #checks that the feature vector is the correct length 
        self.assertEqual(len(color_features), 72)

        #checks that the sum of the normalizaion factor is 1(since its normalized)
        self.assertAlmostEqual(np.sum(color_features), 1.0)

        #clean test image file
        import os 
        os.remove(test_image_filename)

if __name__ == '__main__':
    unittest.main()        

