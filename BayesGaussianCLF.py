import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional
from mlxtend.plotting import plot_decision_regions


class BayesGaussianCLF:
    def __init__(self):
        # classes priority's
        self.priors = None
        # arrays of classes mean values and std's
        self.x_mean = None
        self.x_std = None

    def fit(self, x: pd.DataFrame, y: pd.Series) -> None:
        classes, cls_count = np.unique(y, return_counts=True)
        n_classes = len(classes)
        self.priors = cls_count / len(y)

        # calculating mean and standard deviations of features by classes
        self.x_mean = np.array([np.mean(x[y == cl], axis=0) for cl in range(n_classes)])
        self.x_std = np.array([np.std(x[y == cl], axis=0) for cl in range(n_classes)])

    # calculate the probability density of the feature according to the Gaussian distribution
    @staticmethod
    def pdf(x: float, mean: float, std: float) -> float:
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def predict(self, x: pd.DataFrame) -> np.array:
        pdfs = np.array([BayesGaussianCLF().pdf(X, self.x_mean, self.x_std) for X in x])
        posteriors = self.priors * np.prod(pdfs, axis=2)  # shorten Bayes formula

        return np.argmax(posteriors, axis=1)


# adding function for drawning graphics
def decision_boundary_plot(x: pd.DataFrame, y: pd.Series,
                           x_train: pd.DataFrame, y_train: pd.Series,
                           clf: BayesGaussianCLF, feature_indexes: list[int], title: Optional[str]=None):
    feature1_name, feature2_name = x.columns[feature_indexes]
    X_feature_columns = x.values[:, feature_indexes]
    X_train_feature_columns = x_train[:, feature_indexes]
    clf.fit(X_train_feature_columns, y_train)

    plot_decision_regions(X=X_feature_columns, y=y.values, clf=clf)
    plt.xlabel(feature1_name)
    plt.ylabel(feature2_name)
    plt.title(title)
