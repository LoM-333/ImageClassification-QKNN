import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops

# Calculating Texture Features with GLCM
# four directions (param angles)
def compTextureFeatures(image, distances = [1], angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]):
    glcm = graycomatrix(image, distances=distances, angles=angles)
    features = {
        'contrast': [],
        'correlation': [],
        'energy': [],
        'entropy': []
    }

    for angle in range(len(angles)):

        contrast = graycoprops(glcm, 'contrast')[0, angle]

        correlation = graycoprops(glcm, 'correlation')[0, angle]

        energy = graycoprops(glcm, 'energy')[0, angle]

        glcmMatrix = glcm[:, :, 0, angle]
        glcmMatrixNorm = glcmMatrix / np.sum(glcmMatrix)
        entropy = -np.sum(glcmMatrixNorm * np.log2(glcmMatrixNorm + (glcmMatrix == 0)))

        features['contrast'].append(contrast)
        features['correlation'].append(correlation)
        features['energy'].append(energy)
        features["entropy"].append(entropy)

    return features

def vectorize_texture_features(imagePath, normalized=True):
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


    features = compTextureFeatures(gray_image)

    # Calculate Variances
    contrastVariance = np.var(features['contrast'])
    correlationVariance = np.var(features['correlation'])
    energyVariance = np.var(features['energy'])
    entropyVariance = np.var(features['entropy'])

    # Calculate Means
    contrastMean = np.mean(features['contrast'])
    correlationMean = np.mean(features['correlation'])
    energyMean = np.mean(features['energy'])
    entropyMean = np.mean(features['entropy'])
    
    textureVector = [
        contrastMean, correlationMean, energyMean, entropyMean,
        contrastVariance, correlationVariance, energyVariance, entropyVariance
    ]

    if not normalized:
        return textureVector
    else:
        norm = np.linalg.vector_norm(textureVector)
        if norm == 0:
            return np.array([0] * 8)
        else:
            return textureVector / norm


