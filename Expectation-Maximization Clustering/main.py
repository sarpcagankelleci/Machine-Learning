import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-5.0, +0.0],
                        [+0.0, +5.0],
                        [+5.0, +0.0],
                        [+0.0, -5.0]])
group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter=",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 4

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):

    centroids = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")
    

    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    
    
    assignment = np.argmin(distances, axis=1)
    

    means = np.zeros((K, X.shape[1]))
    
    
    covariances = np.zeros((K, X.shape[1], X.shape[1]))
    
    priors = np.zeros(K)
    
    for i in range(K):


        means[i] = np.mean(X[assignment == i], axis=0)

        covariances[i] = np.cov(X[assignment == i], rowvar=False) if len(X[assignment == i]) > 1 else np.eye(X.shape[1])

        priors[i] = len(X[assignment == i]) / X.shape[0]



    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    N = X.shape[0]
    D = X.shape[1]
    R = np.zeros((N, K))
    
    for iteration in range(100):
        
        for k in range(K):
            
            
            likelihood = stats.multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
            R[:, k] = priors[k] * likelihood
        
        R /= R.sum(axis=1, keepdims=True)
        
    
        for k in range(K):
            Nk = R[:, k].sum()
  
            means[k] = (R[:, k] @ X) / Nk

            diff = X - means[k]
            
            covariances[k] = (R[:, k][:, np.newaxis] * diff).T @ diff / Nk
       
            priors[k] = Nk / N
    
 
    assignments = np.argmax(R, axis=1)

    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):

    colors = ['red', 'blue', 'green', 'purple']
    

    plt.figure(figsize=(8, 8))
    
    for k in range(K):
        cluster_points = X[assignments == k]
        
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=10, color=colors[k], label=f'Cluster {k+1}')
    
    for k in range(K):
        x, y = np.meshgrid(
            np.linspace(-8, 8, 200), 
            np.linspace(-8, 8, 200)
        )
        
        pos = np.dstack((x, y))
        rv = stats.multivariate_normal(mean=group_means[k], cov=group_covariances[k])
        plt.contour(x, y, rv.pdf(pos), levels=[0.01], colors="black", linestyles='dashed')

    for k in range(K):
        x, y = np.meshgrid(
            np.linspace(-8, 8, 200), 
            np.linspace(-8, 8, 200)
        )
        pos = np.dstack((x, y))
        rv = stats.multivariate_normal(mean=means[k], cov=covariances[k])
        
        
        plt.contour(x, y, rv.pdf(pos), levels=[0.01], colors=colors[k], linestyles='solid')
    
    plt.xlabel('x1')
    
    plt.ylabel('x2')
    
    plt.title('EM Clustering Q4')
    plt.show()
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)