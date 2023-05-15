import os
import pathlib

import numpy as np
from sklearn.datasets import make_classification
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler


def choose_dataset(choice, random_state: int):
    if choice == "Synthetic dataset":
        return __use_synthetic_dataset(random_state)
    else:
        return __use_real_dataset()


# 1000 samples with 8 informative features and 2 classes (binary problem)
def __use_synthetic_dataset(random_state: int):
    return make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        flip_y=0.08,
        random_state=random_state
    )


def __use_real_dataset():
    # Initialize the LabelEncoder and SimpleImputer
    encoder = LabelEncoder()
    imputer = SimpleImputer(strategy='mean')

    cwd = os.getcwd()
    path = pathlib.Path(cwd + '\\datasets\\titanic.csv')
    data = pd.read_csv(path)

    # Drop the non-important features and labels
    X = data.drop(['Survived', 'PassengerId', 'Name'], axis=1)
    y = data['Survived']

    # Encode the categorical features
    X = pd.get_dummies(X, dtype=int)

    # Retrieve all features names
    X_columns_names = X.columns.tolist()

    # Impute the missing values
    X = imputer.fit_transform(X)

    # Select k best features
    selector = SelectKBest(score_func=f_classif, k=10)
    X_new = selector.fit_transform(X=X, y=y)

    # Save selected feature names to .npy file
    np.save('selected_features.npy', np.array(X_columns_names)[selector.get_support()])

    # Encode the labels
    y = encoder.fit_transform(y)

    return X_new, y
