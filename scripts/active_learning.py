import math

import numpy as np
from matplotlib import pyplot as plt
from modAL.models import ActiveLearner
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold

from datasets.dataset_creator import choose_dataset
from methods.estimators import choose_estimator

# Our own least_confidence method
from methods.my_least_confidence import least_confidence_sampling
from methods.utilities import plot_scores, plot_accuracy, get_seed


def run_active_learning(choices):
    # Choose a dataset
    X_raw, y_raw = choose_dataset(choices['dataset'], get_seed())

    # Plot raw dataset
    # plt.figure(figsize=(8.5, 6), dpi=130)
    # plt.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw)
    # plt.title('Raw dataset')

    # Define RepeatedStratifiedKFold with 2 splits and 5 repeats
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=get_seed())

    # Define vectors for accuracy scores - noqueried (no pooling) and queried (pooling)
    noqueried_vector = []
    queried_vector = []

    # Define a list for accuracy history of each fold
    accuracy_history = []

    # Create a RepeatedStratifiedKFold loop
    for i, (train_index, test_index) in enumerate(rskf.split(X_raw, y_raw)):
        # Define train and test sets
        X_train, X_test = X_raw[train_index], X_raw[test_index]
        y_train, y_test = y_raw[train_index], y_raw[test_index]

        # Find 5 random indexes for new ground true train subset
        n_labeled_examples = X_train.shape[0]
        training_indices = np.random.randint(low=0, high=n_labeled_examples, size=50)

        # From train set, create a new ground true train subset
        X_train_new = X_train[training_indices]
        y_train_new = y_train[training_indices]

        # From train set, create a new pooling subset
        X_pool = np.delete(X_train, training_indices, axis=0)
        y_pool = np.delete(y_train, training_indices, axis=0)

        # Learner initialization with ground true train subset and Least of Confidence sampling method
        learner = ActiveLearner(
            estimator=choose_estimator(choices['estimator'], __get_n_neighbours(choices, 'n_neighbors')),
            query_strategy=least_confidence_sampling,
            X_training=X_train_new,
            y_training=y_train_new
        )

        # Make a prediction and check if it was correct (for scatter purposes only)
        predictions = learner.predict(X_test)
        before_is_correct = (predictions == y_test)

        # Calculate and report our model's accuracy, then append it to the noqueried_vector
        unqueried_score = accuracy_score(y_test, predictions)
        print('Stats for fold {i}:\n- Accuracy before queries: {acc:0.4f}'.format(i=i + 1, acc=unqueried_score))
        noqueried_vector.append(unqueried_score)

        # Set up a budget (30% of pooling subset)
        budget = round(0.3 * X_pool.shape[0])
        # print('Budget: ', budget)
        performance_history = [unqueried_score]

        # Create a ActiveLearning loop
        for index in range(budget):
            # Query a pooling subset
            query_index, query_instance = learner.query(X_pool)

            # Teach our ActiveLearner model the record it has requested.
            X, y = X_pool[query_index], y_pool[query_index]
            learner.teach(X=X, y=y)

            # Remove the queried instance from the unlabeled pooling subset.
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

            # Predict and calculate model's accuracy.
            queried_score = learner.score(X_test, y_test)

            # Save our model's performance for plotting.
            performance_history.append(queried_score)

        accuracy_history.append(performance_history)

        print(
            '- Accuracy after query {n}: {acc:0.4f}'.format(n=budget, acc=performance_history[-1]))

        # Append last model's performance instance to the queried_vector
        queried_vector.append(performance_history[-1])

        # Make a prediction and check if it was correct (for scatter purposes only)
        predictions = learner.predict(X_test)
        after_is_correct = (predictions == y_test)

        print("- Classification report:\n" + classification_report(y_test, predictions))

        # Scatter after-pooling classification
        if __get_want_plots(choices, 'plots'):
            plot_scores(X_test, i, before_is_correct, after_is_correct, unqueried_score, performance_history[-1],
                        budget)

    # Print mean and std of no-pooling accuracy scores
    print("\nUnqueried vector data")
    print("Mean ", round(np.average(noqueried_vector), 3))
    print("Std ", round(np.std(noqueried_vector), 3))

    # Print mean and std of after-pooling accuracy scores
    print("\nQueried vector data")
    print("Mean ", round(np.average(queried_vector), 3))
    print("Std ", round(np.std(queried_vector), 3))

    np.save('queried_vector_data.npy', queried_vector)
    np.save('no_queried_vector_data.npy', noqueried_vector)

    # Plot accuracy history
    plot_accuracy(accuracy_history, choices)
    plt.show()


def __get_n_neighbours(d, key, default=3):
    try:
        return int(d[key])
    except KeyError:
        return default


def __get_want_plots(d, key, default=False):
    try:
        return bool(d[key])
    except KeyError:
        return default
