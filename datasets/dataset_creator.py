import os

from sklearn.datasets import make_classification
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
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
    # Load the data
    data = pd.read_csv(cwd + '\\datasets\\titanic.csv')

    # Drop the non-important features and labels
    X = data.drop(['Survived', 'PassengerId'], axis=1)
    y = data['Survived']

    # Encode the categorical features
    X = pd.get_dummies(X, dtype=int)

    # Impute the missing values
    X = imputer.fit_transform(X)

    # Select k best features
    X_new = SelectKBest(chi2, k=500).fit_transform(X=X, y=y)

    # Encode the labels
    y = encoder.fit_transform(y)

    return X_new, y
