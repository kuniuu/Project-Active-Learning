import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from Method import UncertaintySampling

def run_experiment(X, y, model, query_strategy, budget):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    initial_accuracy = accuracy_score(y_test, y_pred)

    X_pool = X_train.copy()
    y_pool = y_train.copy()

    for i in range(budget):
        query_idx, query_instance = query_strategy.query(X_pool)
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx)

        model.fit(X_pool, y_pool)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

    return initial_accuracy, accuracy

X, y = make_classification(n_samples=1000, n_features=20,
                            n_informative=2, n_redundant=2,
                            random_state=42)

model = LogisticRegression(random_state=42)
query_strategy = UncertaintySampling(model=model, budget=10)

initial_accuracy, accuracy = run_experiment(X=X,
                                             y=y,
                                             model=model,
                                             query_strategy=query_strategy,
                                             budget=10)

print(f'Initial accuracy: {initial_accuracy:.2f}')
print(f'Accuracy after 10 queries: {accuracy:.2f}')
