import unittest
import os
import numpy as np
import cv2
from src.texture_feature import vectorize_texture_features, normed_textureVector

class TextureFeatureTest(unittest.TestCase):

    def texture_test(self):

        # Create a test image (solid red color)
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[:] = [0, 0, 255]  # Red in BGR

        # Save the test image to a temporary file
        test_image_filename = 'test_image.jpg'
        cv2.imwrite(test_image_filename, test_image)

        unnormed = vectorize_texture_features(test_image_filename)
        print(f"Un-normalized texture vector for solid red image: {unnormed}")
        # values for constant image (https://web.pdx.edu/~jduh/courses/Archive/geog481w07/Students/Hayes_GreyScaleCoOccurrenceMatrix.pdf)
        self.assertAlmostEqual(unnormed[0], 1)
        self.assertEqual(unnormed[1], np.nan)
        self.assertAlmostEqual(unnormed[2], 1)
        self.assertAlmostEqual(unnormed[4], 0)
        self.assertAlmostEqual(unnormed[6], 0)

        # Clean up the test image file
        os.remove(f"../{test_image_filename}")

        img_paths = os.listdir("test_images")
        for i in img_paths:
            normed_vec = normed_textureVector(f"test_images/{i}")
            print(f"Normalized texture vector for {i}: {normed_vec}")
            self.assertEqual(len(normed_vec), 8)
            self.assertAlmostEqual(np.sum(normed_vec), 1.0)
                



if __name__ == '__main__':
    unittest.main()