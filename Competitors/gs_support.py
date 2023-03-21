# This code has been based on Laugel et al. (2018). Full reference below:
# Laugel, T., Lesot, M. J., Marsala, C., Renard, X., & Detyniecki, M. (2017). Inverse classification for comparison-based interpretability in machine learning. arXiv preprint arXiv:1712.08443.
# Code may be found at GitHub repository:
# https://github.com/thibaultlaugel/growingspheres/blob/master/growingspheres/growingspheres.py

import numpy as np
from scipy.stats import kendalltau
from sklearn.metrics.pairwise import pairwise_distances
    
def get_distances(x1, x2, metrics=None):
    x1, x2 = x1.reshape(1, -1), x2.reshape(1, -1)
    euclidean = pairwise_distances(x1, x2)[0][0]
    same_coordinates = sum((x1 == x2)[0])
    
    #pearson = pearsonr(x1, x2)[0]
    kendall = kendalltau(x1, x2)
    out_dict = {'euclidean': euclidean,
                'sparsity': x1.shape[1] - same_coordinates,
                'kendall': kendall
               }
    return out_dict        

def generate_ball(center, r, n):
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[0]
    u = np.random.normal(0,1,(n, d+2))  # an array of (d+2) normally distributed random variables
    norm_ = norm(u)
    u = 1/norm_[:,None]* u
    x = u[:, 0:d] * r #take the first d coordinates
    x = x + center
    return x

def generate_ring(center, segment, n):
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[0]
    z = np.random.normal(0, 1, (n, d))
    try:
        u = np.random.uniform(segment[0]**d, segment[1]**d, n)
    except OverflowError:
        raise OverflowError("Dimension too big for hyperball sampling. Please use layer_shape='sphere' instead.")
    r = u**(1/float(d))
    z = np.array([a * b / c for a, b, c in zip(z, r,  norm(z))])
    z = z + center
    return z

def generate_sphere(center, r, n):    
    def norm(v):
            return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[0]
    z = np.random.normal(0, 1, (n, d))
    z = z/(norm(z)[:, None]) * r + center
    return z