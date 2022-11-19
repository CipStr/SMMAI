import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.sparse.linalg as la
import skimage as sk


def PCA(X, Y, k, digits, plot=True):
    # compute the mean of the dataset
    mean = np.mean(X_train, axis=1)
    # compute the centered dataset
    X_centered = X - mean[:, None]
    print(f"Shape of X_train_centered: {X_centered.shape}")
    # Compute the centered version of as X_train = X_train - X_train_centered , where the subtraction between matrix and
    # vector is executed column-by-column;
    # Compute the SVD of X_train_centered
    U, S, V = np.linalg.svd(X_centered, full_matrices=False)
    # given k compute the truncated svd of X_train_centered
    U_k = U[:, :k]
    # project the dataset
    Z = U_k.T @ X
    print(f"Shape of Z_train: {Z.shape}")
    # visualise the clusters
    plt.scatter(Z[0, :], Z[1, :], c=Y)
    if plot:
        # save the figure
        plt.savefig('pca_projection.png')
    # print position of centroids for each cluster
    for i in range(len(digits)):
        print(f"Position of centroids for each cluster (PCA): {U_k.T @ np.mean(X[:, Y == digits[i]], axis=1)}")
    return U_k


def LDA(X, Y, k, digits, plot=True):
    # compute the mean of the dataset
    mean = np.mean(X, axis=1)
    # compute the mean of each class
    mean_class = np.zeros((X.shape[0], len(digits)))
    # compute the centered dataset of each class
    X_centered = np.zeros(X.shape)
    for i in range(len(digits)):
        mean_class[:, i] = np.mean(X[:, Y == digits[i]], axis=1)
        # compute centered dataset in each class
        X_centered[:, Y == digits[i]] = X[:, Y == digits[i]] - mean_class[:, i][:, np.newaxis]
    # compute the within-class scatter matrix
    S_w = X_centered @ X_centered.T
    # compute the X_bar matrix for each mean class
    X_bar_1 = np.repeat(mean_class[:, 0][:, np.newaxis], X.shape[1], axis=1)
    X_bar_2 = np.repeat(mean_class[:, 1][:, np.newaxis], X.shape[1], axis=1)
    X_bar_3 = np.repeat(mean_class[:, 2][:, np.newaxis], X.shape[1], axis=1)
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
    _, W = la.eigs(np.linalg.inv(L) @ S_b @ L, k)
    W = np.real(W)

    # Compute Q
    Q = np.linalg.inv(L).T @ W
    # project the dataset
    Z_lda = Q.T @ X
    # visualise the clusters
    plt.scatter(Z_lda[0, :], Z_lda[1, :], c=Y)
    # save the figure
    if plot:
        plt.savefig('lda_projection.png')
    # get coordinates of centroids for each cluster and print them in regard to the projection
    for i in range(len(digits)):
        print(f"Position of centroids for each cluster (LDA): {Q.T @ mean_class[:, i]}")
    return Q


# define a classification algorithm that given a new observation x computes the distance from the centroid of each
# cluster and returns the label of the cluster with the smallest distance
def classify(x, centroids):
    # compute the distance from the centroid for each cluster
    dist = np.linalg.norm(x - centroids, axis=0)
    # return the label of the cluster with the smallest distance
    return np.argmin(dist)

# load the dataset data.csv into a pandas dataframe and eplore the head and shape
df = pd.read_csv('data.csv')
print(df.head())
print("Shape of dataframe", df.shape)

# split the dataset into X and Y, X has dimensions (784, 42000) and Y has dimensions (42000,) containing the labels
X = df.iloc[:, 1:].values.T
Y = df.iloc[:, 0].values
print("Shape of X", X.shape)
print("Shape of Y", Y.shape)

# for digits (0,6,9) extract the corresponding rows from X and Y
digits = [0, 6, 9]  # digits to consider
# inline for loop used to filter the dataset
X = X[:, np.isin(Y, digits)]
Y = Y[np.isin(Y, digits)]
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

N_train = int(X.shape[1] * 0.8)
N = X.shape[1]
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

# PCA
U_k = PCA(X, Y, 2, digits)
# LDA
Q = LDA(X, Y, 2, digits)

# for both pca and lda compute for each cluster the average distance from the centroid
# compute the distance from the centroid for each cluster
for i in range(len(digits)):
    # get the indices of the cluster
    idx = np.where(Y_train == digits[i])
    # compute the distance from the centroid for each point
    dist_pca = np.linalg.norm(X_train[:, idx] - U_k.T @ np.mean(X_train[:, idx], axis=1), axis=0)
    # compute the average distance from the centroid
    avg_dist_pca = np.mean(dist_pca)
    print(f"Average distance from the centroid for cluster {digits[i]} (PCA): {avg_dist_pca}")
    dist_lda = np.linalg.norm(X_train[:, idx] - Q.T @ np.mean(X_train[:, idx], axis=1), axis=0)
    avg_dist_lda = np.mean(dist_lda)
    print(f"Average distance from the centroid for cluster {digits[i]} (LDA): {avg_dist_lda}")

for i in range(len(digits)):
    # get the indices of the cluster
    idx = np.where(Y_test == digits[i])
    # compute the distance from the centroid for each point
    dist_pca = np.linalg.norm(X_test[:, idx] - U_k.T @ np.mean(X_test[:, idx], axis=1), axis=0)
    # compute the average distance from the centroid
    avg_dist_pca = np.mean(dist_pca)
    print(f"Average distance from the centroid for cluster {digits[i]} (PCA): {avg_dist_pca}")
    dist_lda = np.linalg.norm(X_test[:, idx] - Q.T @ np.mean(X_test[:, idx], axis=1), axis=0)
    avg_dist_lda = np.mean(dist_lda)
    print(f"Average distance from the centroid for cluster {digits[i]} (LDA): {avg_dist_lda}")


# compute the accuracy of the classification algorithm for both PCA and LDA on the test set
# compute the centroids for each cluster
centroids_pca = U_k.T @ np.mean(X_test, axis=1)
centroids_lda = Q.T @ np.mean(X_test, axis=1)
# compute the accuracy of the classification algorithm for PCA
acc_pca = 0
for i in range(X_test.shape[1]):
    if classify(U_k.T @ X_test[:, i], centroids_pca) == Y_test[i]:
        acc_pca += 1
acc_pca /= X_test.shape[1]
print(f"Accuracy of the classification algorithm for PCA: {acc_pca}")
# compute the accuracy of the classification algorithm for LDA
acc_lda = 0
for i in range(X_test.shape[1]):
    if classify(Q.T @ X_test[:, i], centroids_lda) == Y_test[i]:
        acc_lda += 1
acc_lda /= X_test.shape[1]
print(f"Accuracy of the classification algorithm for LDA: {acc_lda}")

# consider an image from skimage library
img = sk.data.camera()
# compute the SVD of the image
U, s, V = np.linalg.svd(img)
# visualize the dyad of the SVD decomposition
plt.clf()
plt.imshow(U @ np.diag(s) @ V, cmap='gray')
plt.savefig('dyad.png')
plt.clf()
# plot each value of s on a log scale
plt.plot(np.log(s))
plt.savefig('singular_values.png')
plt.clf()
# visualize the k-th approximation of the image for k = 1, 2, 3, 4, 5, 10, 20, 50, 100, 200
for k in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]:
    plt.imshow(U[:, :k] @ np.diag(s[:k]) @ V[:k, :], cmap='gray')
    plt.savefig(f'approximation_{k}.png')
    plt.clf()

# compute and plot the approximation error for k = 1, 2, 3, 4, 5, ..., 200
error = []
for k in range(1, 201):
    error.append(np.linalg.norm(img - U[:, :k] @ np.diag(s[:k]) @ V[:k, :]))
plt.plot(np.log(error))
plt.savefig('error.png')
plt.clf()

# plot the compression factor for k = 1, 2, 3, 4, 5, ..., 200
compression_factor = []
for k in range(1, 201):
    compression_factor.append(img.size / (k + k * img.shape[0]))
plt.plot(np.log(compression_factor))
plt.savefig('compression_factor.png')
