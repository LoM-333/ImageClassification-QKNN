import numpy as np
from color_feature import *
from texture_feature import *

def combine_color_texture_features(FILENAME: str, normalized=True) -> np.ndarray:
    color_features = extract_color_features(FILENAME, normalized=False)
    texture_features = vectorize_texture_features(FILENAME, normalized=False)
    result = np.concatenate((color_features, texture_features))
    if normalized:
        norm = np.linalg.norm(result)
        if norm == 0:
            return result
        else:
            return result / norm
    return result

if __name__ == '__main__':
    print(combine_color_texture_features("tests/test_images/garfield.jpg"))