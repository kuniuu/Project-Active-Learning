import numpy
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.naive_bayes import GaussianNB

X_raw, y_raw = make_classification(
    n_samples=400,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    flip_y=0.08
)

plt.figure(figsize=(8.5, 6), dpi=130)
plt.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw)
plt.title('Raw dataset')
plt.show()

rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)
unqueried_vector = []
queried_vector = []

for i, (train_index, test_index) in enumerate(rskf.split(X_raw, y_raw)):
    # wydzielenie pierwotnego zbioru treningowego i testowego przez walidację krzyżową
    X_train, X_test = X_raw[train_index], X_raw[test_index]
    y_train, y_test = y_raw[train_index], y_raw[test_index]

    # wydzielenie podzbioru uczącego i podzbioru do pool'ingu z pierwotnego zbioru treningowego
    n_labeled_examples = X_train.shape[0]
    training_indices = numpy.random.randint(low=0, high=n_labeled_examples, size=5)

    # podzbior uczacy
    X_train_new = X_train[training_indices]
    y_train_new = y_train[training_indices]

    # podzbior do pool'ingu
    X_pool = numpy.delete(X_train, training_indices, axis=0)
    y_pool = numpy.delete(y_train, training_indices, axis=0)

    # inicjalizacja learnera
    learner = ActiveLearner(
        estimator=GaussianNB(),
        query_strategy=uncertainty_sampling,
        X_training=X_train_new,
        y_training=y_train_new
    )

    # warunek logiczny do scatterowania
    predictions = learner.predict(X_test)
    is_correct = (predictions == y_test)

    # robimy predykcje i sprawdzamy accuracy naszego modelu bez pool'owania
    unqueried_score = learner.score(X_test, y_test)
    print('\nAccuracy before queries in repeat {i}: {acc:0.4f}'.format(i=i + 1, acc=unqueried_score))
    unqueried_vector.append(unqueried_score)

    # scatterujemy predykcje klas bez pool'ingu
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    ax.scatter(X_test[:, 0][is_correct], X_test[:, 1][is_correct], c='g', marker='+', label='Correct', alpha=8/10)
    ax.scatter(X_test[:, 0][~is_correct], X_test[:, 1][~is_correct], c='r', marker='x', label='Incorrect', alpha=8/10)
    ax.legend(loc='lower right')
    ax.set_title('ActiveLearner class prediction without pooling in repeat {i} (Accuracy: {score:.3f})'.format(
        i=i + 1, score=unqueried_score))
    plt.show()

    # set up budget
    budget = round(0.3 * X_pool.shape[0])
    print('Budget: ', budget)
    performance_history = [unqueried_score]

    for index in range(budget):
        query_index, query_instance = learner.query(X_pool)

        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index], y_pool[query_index]
        learner.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pool.
        X_pool, y_pool = numpy.delete(X_pool, query_index, axis=0), numpy.delete(y_pool, query_index)

        # Calculate and report our model's accuracy.
        queried_score = learner.score(X_raw, y_raw)
        print('Accuracy after query {n} in repeat {i}: {acc:0.4f}'.format(n=index + 1, i=i + 1, acc=queried_score))

        # Save our model's performance for plotting.
        performance_history.append(queried_score)

    queried_vector.append(performance_history[-1])

    # aktualizacja warunku logicznego do scatterowania
    predictions = learner.predict(X_test)
    is_correct = (predictions == y_test)

    # scatterujemy predykcje klas po pool'ingu
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    ax.scatter(X_test[:, 0][is_correct], X_test[:, 1][is_correct], c='g', marker='+', label='Correct', alpha=8 / 10)
    ax.scatter(X_test[:, 0][~is_correct], X_test[:, 1][~is_correct], c='r', marker='x', label='Incorrect', alpha=8 / 10)
    ax.legend(loc='lower right')
    ax.set_title('ActiveLearner class prediction after {n} queries in repeat {i} (Accuracy: {final_acc:.3f})'.format(
        n=budget, i=i + 1, final_acc=performance_history[-1]))
    plt.show()

print("\nUnqueried vector data")
print("Mean ", round(numpy.average(unqueried_vector), 3))
print("Std ", round(numpy.std(unqueried_vector), 3))

print("\nQueried vector data")
print("Mean ", round(numpy.average(queried_vector), 3))
print("Std ", round(numpy.std(queried_vector), 3))
