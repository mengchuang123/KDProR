import numpy as np
from scipy.spatial.distance import cosine

def compute_intersection_KNN_fine_grained(indices_1, indices_2, B):
    intersection_matrix_float = np.zeros((B, B), dtype=float)
    for i in range(B):
        for j in range(B):
            if not set(indices_1[i]).isdisjoint(indices_2[j]):
                intersection_matrix_float[i, j] = 1.0
    return intersection_matrix_float


def compute_intersection_KNN_coarse_grained(features_1, features_2, B, top_k=5, threshold=0.7):

    similarity_matrix = np.zeros((B, B), dtype=float)

    for i in range(B):
        for j in range(B):
            for vec1 in features_1[i]:
                for vec2 in features_2[j]:
                    cos_dist = cosine(vec1, vec2)
                    if 1 - cos_dist > threshold:
                        similarity_matrix[i, j] = 1.0
                        break  
                if similarity_matrix[i, j] == 1.0:
                    break  

    return similarity_matrix


