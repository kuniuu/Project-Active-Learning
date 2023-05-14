from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def choose_estimator(choice, n_neighbors):
    if choice is "GaussianNB":
        estimator = GaussianNB()
        return estimator
    elif choice is "kNearestNeighbours":
        estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
        return estimator
    else:
        estimator = DecisionTreeClassifier()
        return estimator
