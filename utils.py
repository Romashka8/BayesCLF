import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


# ----------------------------------------------------------------------------------------------------------------------------

def decision_boundary_plot(x, y, x_train, y_train, clf, feature_indexes, title):

	"""
	Функция для удобного постоения решающих границ Байесовского классификатора.
	"""

	feature_name1, feature_name2 = x.columns[feature_indexes]
	X_feature_columns = x.values[:, feature_indexes]
	X_train_feature_columns = x_train[:, feature_indexes]

	clf.fit(X_train_feature_columns, y_train)

	plot_decision_regions(X=X_feature_columns, y=y.values, clf=clf)
	
	plt.xlabel(feature_name1)
	plt.ylabel(feature_name2)
	plt.title(title)

# ----------------------------------------------------------------------------------------------------------------------------
