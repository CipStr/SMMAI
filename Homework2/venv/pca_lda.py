import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse.linalg as la

# The task for this exercise is to compare PCA and LDA in their ability to cluster when projecting very
# high-dimensional datapoints to 2 or 3 dimensions. In particular, consider the dataset MNIST provided on
# Virtuale. This dataset contains images of handwritten digits with dimension 28×28, together with a number
# from 0 to 9 representing the label. You are asked to:

# Load the dataset in memory and explore its head and shape to understand how the informations are
# placed inside it;

dataset = pd.read_csv('data.csv')
print(dataset.head())
print(f"Dataset shape: {dataset.shape}")

# Split the dataset into the X matrix of dimension d × N , with d = 784 being the dimension of each
# datum, N is the number of datapoints, and Y ∈ RN containing the corresponding labels;

X = np.array(dataset.iloc[:, 1:]).T
Y = np.array(dataset.iloc[:, 0])
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

# Choose a number of digits (for example, 0, 6 and 9) and extract from X and Y the sub-dataset
# containing only the considered digits. Re-call X and Y those datasets, since the originals are not
# required anymore;

digits = [1, 2, 3]  # digits to consider
# inline for loop used to filter the dataset
X = X[:, np.isin(Y, digits)]
Y = Y[np.isin(Y, digits)]
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

# Set Ntrain < N and randomly sample a training set with N_train datapoints from X (and the
# corresponding Y ). Call them X_train and Y_train. Everything else is the test set. Call it Xtest and
# Ytest.

N_train = 1000
N = X.shape[1]  # 13212
# random permutation of the indices of the dataset
idx = np.random.choice(N, N_train, replace=False)
X_train = X[:, idx]
Y_train = Y[idx]
X_test = np.delete(X, idx, axis=1)
Y_test = np.delete(Y, idx, axis=0)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of Y_train: {Y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of Y_test: {Y_test.shape}")

# Implement the algorithms computing the PCA and LDA of X_train with a fixed k. Visualize the results
# (for k = 2) and the position of the centroid of each cluster;

k = 2
# PCA
# compute the mean of the dataset
mean = np.mean(X_train, axis=1)
# compute the centered dataset
X_train_centered = X_train - mean[:, None]
print(f"Shape of X_train_centered: {X_train_centered.shape}")
# Compute the centered version of as X_train = X_train - X_train_centered , where the subtraction between matrix and
# vector is executed column-by-column;
# Compute the SVD of X_train_centered
U, S, V = np.linalg.svd(X_train_centered, full_matrices=False)
# given k compute the truncated svd of X_train_centered
U_k = U[:, :k]
# project the dataset
Z_train = U_k.T @ X_train
print(f"Shape of Z_train: {Z_train.shape}")
# visualise the clusters
plt.scatter(Z_train[0, :], Z_train[1, :], c=Y_train)
# save the figure
plt.savefig('pca.png')

# LDA
# compute the mean of the dataset
mean = np.mean(X_train, axis=1)
# compute the mean of each class
mean_class = np.zeros((X_train.shape[0], len(digits)))
# compute the centered dataset of each class
X_train_centered = np.zeros(X_train.shape)
for i in range(len(digits)):
    mean_class[:, i] = np.mean(X_train[:, Y_train == digits[i]], axis=1)
    # compute centered dataset in each class
    X_train_centered[:, Y_train == digits[i]] = X_train[:, Y_train == digits[i]] - mean_class[:, i][:, np.newaxis]
# compute the within-class scatter matrix
S_w = X_train_centered @ X_train_centered.T
# compute the X_bar matrix for each mean class
X_bar_1 = np.repeat(mean_class[:, 0][:, np.newaxis], X_train.shape[1], axis=1)
X_bar_2 = np.repeat(mean_class[:, 1][:, np.newaxis], X_train.shape[1], axis=1)
X_bar_3 = np.repeat(mean_class[:, 2][:, np.newaxis], X_train.shape[1], axis=1)
# compute the between-class cluster dataset
X_bar = np.concatenate((X_bar_1, X_bar_2, X_bar_3), axis=1)
X_bar_mean = X_bar - mean[:, np.newaxis]
# compute the between-class scatter matrix
S_b = X_bar_mean @ X_bar_mean.T
# try Cholesky decomposition
try:
    L = np.linalg.cholesky(S_w)
except:
    epsilon = 1e-6
    S_w = S_w + epsilon * np.eye(S_w.shape[0])
    L = np.linalg.cholesky(S_w)

# Compute the first k eigenvector decomposition of L^-1 Sb L
_, W = la.eigs(np.linalg.inv(L) @ S_b @ L, k=2)
W = np.real(W)

# Compute Q
Q = np.linalg.inv(L).T @ W
# project the dataset
Z_train_lda = Q.T @ X_train
# visualise the clusters
plt.scatter(Z_train_lda[0, :], Z_train_lda[1, :], c=Y_train)
# save the figure
plt.savefig('lda.png')

# For both the algorithms, compute for each cluster the average distance from the centroid on the test
# set. Comment the results;
# PCA
# project the test set
Z_test = U_k.T @ X_test



