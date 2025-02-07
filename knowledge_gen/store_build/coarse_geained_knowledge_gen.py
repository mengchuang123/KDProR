import torch
from sklearn.cluster import KMeans
import numpy as np

vector_path = 'FINE GRAINED KNOWLEDGE PATH'

vectors = torch.load(vector_path)

B, dim = vectors.shape
kmeans = KMeans(n_clusters=256, random_state=0).fit(vectors)

labels = kmeans.labels_

cluster_vectors = torch.zeros((256, dim))

for i in range(256):
    cluster = vectors[labels == i]
    
    if len(cluster) > 0:
        # Max pooling
        cluster_vectors[i, :] = torch.max(cluster, dim=0)[0]

new_vector_path = 'COARSE FEATURE PATH'
torch.save(cluster_vectors, new_vector_path)
