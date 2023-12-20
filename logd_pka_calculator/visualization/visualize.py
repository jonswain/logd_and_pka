"""Functions for visualising data."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

sns.set_style("dark")


def plot_distribution(dataframe, column, bins=50, log=False) -> None:
    """Generates a distribution plot for a column in a dataframe

    Args:
        dataframe (pandas dataframe): Pandas dataframe containing the data
        column (string): Column to be plotted
        bins (int): Number of bins for histogram
        log (bool, optional): Log transform data. Defaults to False.
    """
    data = dataframe[column]
    data = data.apply(np.log10) if log else data
    x_label = f"Log({column})" if log else f"{column}"
    # Plot graph
    plt.figure(figsize=(8, 8))
    sns.histplot(data, bins=bins)
    plt.xlabel(f"{x_label}")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {x_label}")
    plt.show()


def compare_train_test_distribution(
    train, test, column, bins=50, name="y_true"
) -> None:
    """Generates a distribution plot for a column in a dataframe for the training
    and test sets

    Args:
        train (pandas dataframe): Pandas dataframe containing the training data
        test (pandas dataframe): Pandas dataframe containing the test data
        column (string): Column to be plotted
        bins (int): Number of bins for histogram
        log (bool, optional): Log transform data. Defaults to False.
    """
    _, axis = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    # Plot train distribution
    train_data = train[column]
    sns.histplot(data=train_data, bins=bins, ax=axis[0])
    axis[0].set_title(f"Distribution of {name} in training set", fontsize=12)
    axis[0].set_xlabel(f"Distribution of {name}", fontsize=9)
    axis[0].set_ylabel("Frequency", fontsize=9)
    # Plot test distribution
    test_Data = test[column]
    sns.histplot(data=test_Data, bins=bins, ax=axis[1])
    axis[1].set_title(f"Distribution of {name} in test set", fontsize=12)
    axis[1].set_xlabel(f"Distribution of {name}", fontsize=9)
    axis[1].set_ylabel("Frequency", fontsize=9)
    plt.show()


def pca_then_tsne(features, classes) -> None:
    """Uses PCA to reduce the features to 50 principal components, then
    futher reduces the dataset to two dimensions using tSNE. Plots all data
    points using these two dimesnions.

    Args:
        features (list[array]): A list of features for each class, each array
        in the list must be the same size.
        labels (list[string]): A list of classes for labelling
    """
    combined_features = pd.concat(features)
    scaler = preprocessing.StandardScaler().fit(combined_features)
    scaled = scaler.transform(combined_features)

    pca = PCA(n_components=50)
    crds = pca.fit_transform(scaled)
    print(
        f"Variance in top 50 principle components: {sum(pca.explained_variance_ratio_):.3f}"
    )
    labels = []
    for i, feature in enumerate(features):
        try:
            labels += [classes[i]] * len(feature)
        except IndexError:
            labels += ["Unknown"] * len(feature)
    crds_embedded = TSNE(n_components=2).fit_transform(crds)

    _, axis = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
    sns.scatterplot(
        x=crds_embedded[:, 0],
        y=crds_embedded[:, 1],
        hue=labels,
        s=10,
        alpha=0.5,
        ax=axis,
        palette=["green", "orange"],
    )
    plt.show()


def regression_model(y_true, y_pred) -> None:
    """Displays a plot showing the performance of a model

    Inputs:
        y_true: True values for target
        y_pred: Predicted target values from model
    Output: Displays plot
    """
    model_metrics = f"""
    R2: {r2_score(y_true, y_pred):.2f},
    MSE: {mean_squared_error(y_true, y_pred):.2f}
    RMSE: {(mean_squared_error(y_true, y_pred, squared=False)):.2f}
    MAE: {mean_absolute_error(y_true, y_pred):.2f}"""

    _ = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    plt.scatter(y_true, y_pred)
    plt.title("Predicted vs Experimental values")
    ax_min = min(np.concatenate([y_true, y_pred], axis=0))
    ax_max = max(np.concatenate([y_true, y_pred], axis=0))
    ax_range = ax_max - ax_min
    plt.xlim(ax_min - (ax_range / 10), ax_max + (ax_range / 10))
    plt.ylim(ax_min - (ax_range / 10), ax_max + (ax_range / 10))
    x = np.linspace(ax_min - (ax_range / 10), ax_max + (ax_range / 10), 1000)
    plt.plot(x, x, "--k")
    plt.xlabel("Experimental values")
    plt.ylabel("Predicted values")
    plt.text(0.1, 0.75, f"{model_metrics}", fontsize=15, transform=ax.transAxes)
    plt.show()
