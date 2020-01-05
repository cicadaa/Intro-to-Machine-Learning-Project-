# exercise 11.1.5
from matplotlib.pyplot import figure, plot, legend, xlabel, show
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn import model_selection

# Load Matlab data file and extract variables of interest
data = pd.read_csv('/Users/cicada/Documents/DTU/ML/Toolbox/02450Toolbox_Python/Data/SAheart.data')
data['famhist'] = data['famhist'].apply(lambda x: x == 'Present').astype(int)
features = ['tobacco', 'adiposity','ldl', 'famhist', 'obesity','alcohol','age','typea']
target = 'chd'

X = data[features].values
y = data[target].values

mu = np.mean(X, 0)
sigma = np.std(X, 0)
X = (X - mu) / sigma

N, M = X.shape
classNames = ['nochd','chd']
C = len(classNames)

# Range of K's to try
KRange = range(1,12)
T = len(KRange)

covar_type = 'diag'       # you can try out 'diag' as well
reps = 3                  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans' # 'kmeans' or 'random'

# Allocate variables
BIC = np.zeros((T,))
AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)
        
        # Get BIC and AIC
        BIC[t,] = gmm.bic(X)
        AIC[t,] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            

# Plot results

figure(1); 
plot(KRange, BIC,'-*b')
plot(KRange, AIC,'-xr')
plot(KRange, 2*CVE,'-ok')
legend(['BIC', 'AIC', 'Crossvalidation'])
xlabel('K')
show()

print('Ran Exercise 11.1.5')