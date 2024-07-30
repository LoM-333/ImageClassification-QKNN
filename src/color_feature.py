import cv2
import numpy as np
import os

def quantize_h(H):
    h = H
    '''
    h[((h >= 0) & (h <= 20)) | ((316 >= 0) & (h <= 359))] = 0
    #[0 <= h <= 20 | 316 <= h <= 359] = 0
    h[((h >= 21) & (h <= 40))] = 1
    h[((h >= 41) & (h <= 75))] = 2
    h[((h >= 76) & (h <= 155))] = 3
    h[((h >= 156) & (h <= 190))] = 4
    h[((h >= 191) & (h <= 270))] = 5
    h[((h >= 271) & (h <= 295))] = 6
    h[((h >= 296) & (h <= 315))] = 7
    '''
    
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
    s[((s >= 0) & (s < 0.2))] = 0
    s[((s >= 0.2) & (s < 0.7))] = 1
    s[((s >= 0.7) & (s < 1))] = 2

    return s

def quantize_b(B):
    b = B
    b[((b >= 0) & (b < 0.2))] = 0
    b[((b >= 0.2) & (b < 0.7))] = 1
    b[((b >= 0.7) & (b < 1))] = 2

    return b

def extract_color_features(FILENAME, normalized=True):

    image = cv2.imread(FILENAME)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsb_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    H, S, B = cv2.split(hsb_image)
  
    # fit to match values in paper
    H = H * (359 / 179)
    S = S / 255
    B = B / 255


    h = quantize_h(H)
    s = quantize_s(S)
    b = quantize_b(B)


    Q_h = 9
    Q_s = 3
    G = h * Q_h + s * Q_s + b
    num_bins = 72

    color_feature_vector, _ = np.histogram(G, bins=num_bins, range=(0, num_bins))
    if not normalized:
        return color_feature_vector
    else:
        norm = np.linalg.norm(color_feature_vector)
        if norm != 0:
            return color_feature_vector / norm
        else:
            return np.array([0] * 72)

if __name__ == "__main__":
    FILENAME = os.path.join(os.getcwd(), "tests", "test_images", "platypus.jpg")
    color_features = extract_color_features(FILENAME)
    print("Color Feature Vector (normalized):", color_features)