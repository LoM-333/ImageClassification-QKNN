import cv2
import numpy as np

def quantize_h(H):
    h = H
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if 0 <= H[i][j] <= 20 or 316 <= H[i][j] <= 359:
                h[i][j] = 0
            elif 21 <= H[i][j] <= 40:
                h[i][j] = 1
            elif 41 <= H[i][j] <= 75:
                h[i][j] = 2
            elif 75 <= H[i][j] <= 155:
                h[i][j] = 3
            elif 156 <= H[i][j] <= 190:
                h[i][j] = 4
            elif 191 <= H[i][j] <= 270:
                h[i][j] = 5
            elif 271 <= H[i][j] <= 295:
                h[i][j] = 6
            elif 296 <= H[i][j] <= 315:
                h[i][j] = 7
    return h

def quantize_s(S):
    s = S
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if 0 <= S[i][j] < 0.2:
                s[i][j] = 0
            elif 0.2 <= S[i][j] < 0.7:
                s[i][j] = 1
            elif 0.7 <= S[i][j] < 1:
                s[i][j] = 2
    return s

def quantize_b(B):
    b = B
    for i in range(B.shape[0]):
        for j in range(B.shape[1]):
            if 0 <= B[i][j] < 0.2:
                b[i][j] = 0
            elif 0.2 <= B[i][j] < 0.7:
                b[i][j] = 1
            elif 0.7 <= B[i][j] < 1:
                b[i][j] = 2
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
    FILENAME = "../tests/test_images/platypus.jpg"
    color_features = extract_color_features(FILENAME)
    print("Color Feature Vector (normalized):", color_features)