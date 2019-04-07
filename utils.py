import numpy as np

import xgboost as xgb
import catboost

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier



def voting(labels):
    return np.argmax(np.bincount(labels))

def get_min_class_size(labels):
    maxlabel = np.max(labels)
    n = [np.sum(labels == i) for i in range(maxlabel + 1)]
    return np.min(n)

def get_sample(n, y):
    maxlabel = np.max(y)
    a = np.arange(0, len(y))
    result = []
    for label in range(maxlabel + 1):
        ids = np.random.choice(a[y == label], n, replace=False)
        result.append(ids)
    return np.hstack(result)

def new_model(n_components, classifier_type):
    if classifier_type == 'dt':
        classifier = DecisionTreeClassifier()
    elif classifier_type == 'rf':
        classifier = RandomForestClassifier()
    elif classifier_type == 'xgb':
        classifier = xgb.XGBClassifier(max_depth=3)
    elif classifier_type == 'knn':
        classifier = KNeighborsClassifier()
    elif classifier_type == 'ctb':
        classifier = catboost.CatBoostClassifier()

    return Pipeline([('pca', PCA(n_components=n_components)), ('classifier', classifier)])
