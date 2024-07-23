import numpy as np
from color_feature import *
from texture_feature import *

def combine_color_texture_features(FILENAME):
    color_features = extract_color_features(FILENAME)
    texture_features = extract_texture_features(FILENAME)
    return np.concatenate((color_features, texture_features))

if __name__ == "__main__":
    FILENAME = "../tests/test_images/platypus.jpg"
    combined_features = combine_color_texture_features(FILENAME)
    print("Combined Features Vector:", combined_features)