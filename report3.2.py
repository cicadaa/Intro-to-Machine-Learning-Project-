# exercise 11.1.1
from matplotlib.pyplot import figure, show
import numpy as np
import pandas as pd
from toolbox_02450 import clusterplot, clusterval

from sklearn.mixture import GaussianMixture

data = pd.read_csv('/Users/cicada/Documents/DTU/ML/Toolbox/02450Toolbox_Python/Data/SAheart.data')
data['famhist'] = data['famhist'].apply(lambda x: x == 'Present').astype(int)
features = ['tobacco', 'adiposity','ldl', 'famhist', 'obesity']#['tobacco', 'adiposity','ldl', 'famhist', 'obesity', 'alcohol','age','typea','sbp']
target = 'chd'
X = data[features].values
y = data[target].values

mu = np.mean(X, 0)
sigma = np.std(X, 0)
X = (X - mu) / sigma
N, M = X.shape
classNames = ['nochd','chd']
C = len(classNames)


# Number of clusters
K = 3
cov_type = 'full' # e.g. 'full' or 'diag'

# define the initialization procedure (initial value of means)
initialization_method = 'random'#  'random' or 'kmeans'
# random signifies random initiation, kmeans means we run a K-means and use the
# result as the starting point. K-means might converge faster/better than  
# random, but might also cause the algorithm to be stuck in a poor local minimum 

# type of covariance, you can try out 'diag' as well
reps = 1
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps, 
                      tol=1e-6, reg_covar=1e-6, init_params=initialization_method).fit(X)
cls = gmm.predict(X)
print(cls)
Rand, Jaccard, NMI,purity = clusterval(y,cls)
# extract cluster labels
cds = gmm.means_        
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])    
    
    count = 0    
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
#figure(figsize=(14,9))
#clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
#show()
print(cds)
## In case the number of features != 2, then a subset of features most be plotted instead.
figure(figsize=(14,9))
idx = [0,1]
# feature index, choose two features to use as x and y axis in the plot
clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
show()

print(Rand, Jaccard, NMI)