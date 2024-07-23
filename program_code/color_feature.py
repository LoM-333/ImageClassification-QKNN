import cv2
import numpy as np

def quantize_h(H):
    h = 0
    if 0 <= H <= 20 or 316 <= H <= 359:
        h = 0
    elif 21 <= H <= 40:
        h = 1
    elif 41 <= H <= 75:
        h = 2
    elif 75 <= H <= 155:
        h = 3
    elif 156 <= H <= 190:
        h = 4
    elif 191 <= H <= 270:
        h = 5
    elif 271 <= H <= 295:
        h = 6
    elif 296 <= H <= 315:
        h = 7
    return h

def quantize_s(S):
    s = 0
    if 0 <= S < 0.2:
        s = 0
    elif 0.2 <= S < 0.7:
        s = 1
    elif 0.7 <= S < 1:
        s = 2
    return s

def quantize_b(B):
    b = 0
    if 0 <= B < 0.2:
        b = 0
    elif 0.2 <= B < 0.7:
        b = 1
    elif 0.7 <= B < 1:
        b = 2
    return b

def extract_color_features(FILENAME):

    image = cv2.imread(FILENAME)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsb_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    H, S, B = cv2.split(hsb_image)
    h = quantize_h(H)
    s = quantize_s(S)
    b = quantize_b(B)

    Q_h = 9
    Q_s = 3
    G = h * Q_h + s * Q_s + b
    num_bins = 72

    color_feature_vector, _ = np.histogram(G, bins=num_bins, range=(0, num_bins))
    normalized_feature_vector = color_feature_vector / np.sum(color_feature_vector)

    return normalized_feature_vector

if __name__ == "__main__":
    FILENAME = ""
    color_features = extract_color_features(FILENAME)
    print("Color Feature Vector (normalized):", color_features)