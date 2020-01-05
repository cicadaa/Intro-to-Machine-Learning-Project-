# report3.1
from matplotlib.pyplot import figure, show
from toolbox_02450 import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import pandas as pd
import numpy as np
from toolbox_02450 import clusterval


# Load Matlab data file and extract variables of interest

data = pd.read_csv('/Users/cicada/Documents/DTU/ML/Toolbox/02450Toolbox_Python/Data/SAheart.data')
data['famhist'] = data['famhist'].apply(lambda x: x == 'Present').astype(int)
features = ['tobacco', 'adiposity','ldl', 'famhist', 'obesity']#,'age' 'alcohol','age','typea']
target = 'chd'

X = data[features].values
y = data[target].values

mu = np.mean(X, 0)
sigma = np.std(X, 0)
X = (X - mu) / sigma

N, M = X.shape
classNames = ['nochd','chd']
C = len(classNames)


# Perform hierarchical/agglomerative clustering on data matrix
Method = 'complete'
Metric = 'euclidean'
#seuclidean，‘braycurtis’, ‘canberra’, ‘chebyshev’,
# ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’,
# ‘euclidean’, ‘hamming’, ‘jaccard’, ‘jensenshannon’,
# ‘kulsinski’, ‘mahalanobis’, ‘matching’, ‘minkowski’,
# ‘rogerstanimoto’, ‘russellrao’, ‘seuclidean’,
# ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’.
Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 3
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
x1 = [x for x in cls if x == 1]
x2 = [x for x in cls if x == 2]
print(len(x1), len(x2))
print(len(data[data.chd == 0]), len(data[data.chd == 1]))
cl = [ int(i-1) for i in cls ]
cl2 = [x for x in cl if x == 0]
cl3 = [x for x in cl if x == 1]
print(len(cl2),len(cl3))
cl = np.array(cl)
Rand, Jaccard, NMI , purity= clusterval(y,cl)
figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=6
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()

print(Rand, Jaccard, NMI,purity)

print('Ran Exercise 10.2.1')