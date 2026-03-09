#%%
# # import data_generate as data_gen
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
from mpl_toolkits.mplot3d import Axes3D 
from statistics import mean, stdev
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import numpy as np

def svd(X, n_components=None):
    """
    X: nparray of shape (batch, n_points, n_dim)
    n_components: number of principal components to keep (None = keep all)

    Returns:
        components: (n_dim, k)
        explained_variance: (k,)
        explained_variance_ratio: (k,)
        mean: (n_dim,)
        projected: (n_points, k)
    """
    batch, n_points, n_dim = X.shape
    X_flat = X.reshape(batch * n_points, n_dim)
    mean = X_flat.mean(axis=0)
    X_centered = X_flat - mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    
    if n_components is None:
        k = n_dim
    else:
        k = n_components

    components = Vt[:k].T  # (n_dim, k)

    return components, S, mean

def reconstruct_from_pca(projected, components, mean):
    batch, n_points, k = projected.shape
    n_dim = components.shape[0]

    projected_flat = projected.reshape(batch * n_points, k)
    X_recon_flat = projected_flat @ components.T + mean
    return X_recon_flat.reshape(batch, n_points, n_dim)

def plot_pca_projection(X, components, ax): # file_name):
    batch, n_points, n_dim = X.shape
    k = components.shape[1]
    X_flat = X.reshape(batch * n_points, n_dim)
    mean = X_flat.mean(axis=0)
    X_centered = X_flat - mean

    projected_flat = X_centered @ components  # (n_points, k)
    projected = projected_flat.reshape(batch, n_points, k)

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')

    for i in range(batch):
        ax.plot(
            projected[i, :, 0],
            projected[i, :, 1],
            projected[i, :, 2]
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_box_aspect([1, 1, 1])

    # plt.tight_layout()
    # plt.savefig(file_name, bbox_inches="tight")
    # plt.show()
    return projected, ax

def plot_explained_variance(S, n_points, ax): # file_name="explained_variance.pdf"):
    """
    S: singular values from SVD
    n_points: number of samples
    """
    # Singular values -> eigenvalues
    eigenvalues = (S**2) / (n_points - 1) # 1 dof lost in computing the mean
    ev_ratio = eigenvalues / eigenvalues.sum()

    # plt.figure()
    ax.plot(ev_ratio)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Explained variance ratio")
    
    # plt.tight_layout()
    # plt.savefig(file_name, format="pdf")
    # plt.show()
    return eigenvalues, ev_ratio, ax

def plot_singular_values(S, ax): # file_name="singular_values.pdf"):

    # plt.figure()
    ax.semilogy(S)  # log scale on y-axis
    ax.set_xlabel("Component index")
    ax.set_ylabel("Singular value (log scale)")
    # plt.title("Singular Value Decay")

    # plt.tight_layout()
    # plt.savefig(file_name, format="pdf")
    # plt.show()
    return ax

def plot_cumulative_variance(S, n_points, ax): # file_name="cumulative_variance.pdf"):

    eigenvalues = (S**2) / (n_points - 1)
    ev_ratio = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(ev_ratio)

    # plt.figure()
    ax.plot(cumulative)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.ylim([0, 1.05])
    # plt.title("Cumulative Explained Variance")

    # plt.tight_layout()
    # plt.savefig(file_name, format="pdf")
    # plt.show()
    return ax

def cross_cov_spectrum(X, Y):
    B, T, d_x = X.shape
    _, _, d_y = Y.shape

    Xf = X.reshape(B*T, d_x)
    Yf = Y.reshape(B*T, d_y)

    Xc = Xf - Xf.mean(axis=0)
    Yc = Yf - Yf.mean(axis=0)

    C = Xc.T @ Yc

    U, S, Vt = np.linalg.svd(C, full_matrices=False)

    return S

def compare_linear_polynomial(X, Y, max_degree=2, test_split=0.2, random_state=0):
    """
    X: (B, T, d_x)
    Y: (B, T, d_y)
    max_degree: maximum polynomial degree
    test_split: fraction used for test set

    Returns:
        dict with train and test R² values
    """

    B, T, d_x = X.shape
    _, _, d_y = Y.shape
    N = B * T

    # Train/test split
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(B)

    n_train = int((1 - test_split) * B)
    n_test = B - n_train
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]

    # Flatten
    X_train, X_test = X_train.reshape(n_train * T, d_x), X_test.reshape(n_test * T, d_x)
    Y_train,  Y_test = Y_train.reshape(n_train * T, d_y), Y_test.reshape(n_test * T, d_y)


    results_train, results_test, n_features = [], [], []

    for degree in range(1, max_degree + 1):

        poly = PolynomialFeatures(degree=degree, include_bias=False)

        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_poly, Y_train)

        Y_train_pred = model.predict(X_train_poly)
        Y_test_pred = model.predict(X_test_poly)

        results_train.append(r2_score(Y_train, Y_train_pred))
        results_test.append(r2_score(Y_test, Y_test_pred))
        n_features.append(X_train_poly.shape[1])

    return np.array(results_train), np.array(results_test), n_features


def r2_linearity_fraction(X, Y, max_degree=2):
    best_r2 = 0
    r2_linear = 0
    for degree in range(1, max_degree+1):
        poly = PolynomialFeatures(degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, Y)
        Y_pred = model.predict(X_poly)
        r2 = r2_score(Y, Y_pred)
        if degree == 1:
            r2_linear = r2
        best_r2 = max(best_r2, r2)
    fraction = r2_linear / best_r2
    return fraction, r2_linear, best_r2
