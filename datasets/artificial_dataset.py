from sklearn.datasets import make_classification


# 400 samples with 2 informative features and 2 classes (binary problem)
def create_dataset(random_state: int):
    return make_classification(
        n_samples=1300,
        n_features=8,
        n_informative=8,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        flip_y=0.08,
        random_state=random_state
    )
