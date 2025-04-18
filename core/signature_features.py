import numpy as np
from skimage.feature import local_binary_pattern, hog
from scipy.ndimage import gaussian_filter
import cv2

class SignatureFeaturesExtractor:
    def __init__(self, image):
        self.image = image

    def extract_all(self):
        """
        Извлекает LBP, кривизну и упрощённые HOG признаки, объединяет в один вектор.
        Также добавлено лёгкое размытие изображения перед извлечением признаков.
        """
        blurred = cv2.GaussianBlur(self.image, (3, 3), 0)

        lbp = self.extract_lbp_features(blurred)
        curve = self.extract_curvature_features(blurred)
        hog_feat = self.extract_hog_features_simple(blurred)
        features = np.concatenate((lbp, curve, hog_feat))
        return features

    def extract_lbp_features(self, image):
        lbp = local_binary_pattern(image, P=8, R=1, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 11), density=True)
        return hist

    def extract_curvature_features(self, image):
        smoothed = gaussian_filter(image.astype(np.float32), sigma=1)
        gx, gy = np.gradient(smoothed)
        gxx, _ = np.gradient(gx)
        _, gyy = np.gradient(gy)
        curvature = np.abs(gxx + gyy)
        stats = [np.mean(curvature), np.std(curvature), np.max(curvature), np.min(curvature)]
        return np.array(stats)

    def extract_hog_features_simple(self, image):
        hog_feat = hog(
            image,
            orientations=4,
            pixels_per_cell=(16, 16),
            cells_per_block=(1, 1),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        return hog_feat
