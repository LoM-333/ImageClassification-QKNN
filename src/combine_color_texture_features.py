import numpy as np
from src.color_feature import *
from src.texture_feature import *

def combine_color_texture_features(FILENAME):
    color_features = extract_color_features(FILENAME)
    texture_features = normed_textureVector(FILENAME)
    return np.concatenate((color_features, texture_features))

