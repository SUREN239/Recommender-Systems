import numpy as np
from scipy.spatial.distance import euclidean, cosine
from scipy.stats import pearsonr

def euclidean_similarity(vec1, vec2):
    return 1 / (1 + euclidean(vec1, vec2))  # Convert distance to similarity

def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)  # Scipy cosine distance is 1 - similarity

def pearson_correlation(vec1, vec2):
    return pearsonr(vec1, vec2)[0]  # Returns correlation coefficient

# Example user-item rating vectors
user1 = np.array([4, 5, 2, 3, 5])
user2 = np.array([5, 3, 4, 2, 4])

print("Euclidean Similarity:", euclidean_similarity(user1, user2))
print("Cosine Similarity:", cosine_similarity(user1, user2))
print("Pearson Correlation:", pearson_correlation(user1, user2))
