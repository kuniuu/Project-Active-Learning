import os

import numpy as np
from modAL.models import ActiveLearner
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RepeatedStratifiedKFold

from datasets.dataset_creator import choose_dataset
from methods.active_learning.estimators import choose_estimator
# Our own least_confidence method
from methods.active_learning.my_least_confidence import least_confidence_sampling
from methods.utils import plot_accuracy, get_seed, save_to_npy, print_mean_and_std, plot_scores, plot_queried_pool


def run_active_learning(simulation_parameters):
    print("\nRunning Active Learning simulation with the following parameters:")
    print("- Estimator: " + simulation_parameters['estimator'])
    print("- Dataset: " + simulation_parameters['dataset'])
    if simulation_parameters['estimator'] == 'kNearestNeighbours':
        print("- n_neighbors: " + simulation_parameters['n_neighbors'])

    # Choose a dataset
    X_raw, y_raw = choose_dataset(simulation_parameters['dataset'], get_seed())

    # Define RepeatedStratifiedKFold with 2 splits and 5 repeats
    rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=get_seed())

    # Define vectors for accuracy scores - before_queries (no pooling) and after_queries (pooling)
    scores_vector = []

    # Define a list for accuracy history of each fold
    accuracy_history = []

    supports_vector = []
    ground_truth_vector = []

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
            estimator=choose_estimator(simulation_parameters['estimator'], __get_n_neighbours(simulation_parameters, 'n_neighbors')),
            query_strategy=least_confidence_sampling,
            X_training=X_train_new,
            y_training=y_train_new
        )

        # Make a prediction and check if it was correct (for scatter purposes only)
        predictions = learner.predict(X_test)
        before_is_correct = (predictions == y_test)

        # Calculate and report our model's accuracy, then append it to the before_queries_scores_vector
        before_queries_score = accuracy_score(y_test, predictions)
        print('\nStats for fold {i}:\n- Accuracy before queries: {acc:0.4f}'.format(i=i + 1, acc=before_queries_score))

        # Set up a budget (30% of pooling subset)
        budget = round(0.3 * X_pool.shape[0])

        performance_history = [before_queries_score]

        # Create arrays for storing queried samples and their labels only for synthetic dataset
        if simulation_parameters['dataset'] == 'synthetic':
            X_queried = np.zeros([budget, 2])
            y_queried = np.zeros([budget])

        # Create a ActiveLearning loop
        for index in range(budget):
            # Query a pooling subset
            query_index, query_instance = learner.query(X_pool)

            if simulation_parameters['dataset'] == 'synthetic':
                X_queried[index] = X_pool[query_index]
                y_queried[index] = y_pool[query_index]

            # Teach our ActiveLearner model the record it has requested.
            X, y = X_pool[query_index], y_pool[query_index]
            learner.teach(X=X, y=y)

            # Remove the queried instance from the unlabeled pooling subset.
            X_pool, y_pool = np.delete(X_pool, query_index, axis=0), np.delete(y_pool, query_index)

            # Predict and calculate model's accuracy.
            queried_score = learner.score(X_test, y_test)

            # Save our model's performance for plotting.
            performance_history.append(queried_score)

            # Plot queried pool only for the first fold as an example, every 50 queries, only for synthetic dataset
            if simulation_parameters['dataset'] == 'synthetic' and (index % 50 == 0 or index == budget - 1) and i == 0:
                plot_queried_pool(X_queried, y_queried.astype(bool), index, i, simulation_parameters)

        accuracy_history.append(performance_history)

        print(
            '- Accuracy after query {n}: {acc:0.4f}'.format(n=budget, acc=performance_history[-1]))

        # Append last model's performance instance to the scores_vector
        scores_vector.append(performance_history[-1])

        # Make a prediction and check if it was correct (for scatter purposes only)
        predictions = learner.predict(X_test)
        probas = learner.predict_proba(X_test)[:, 1]
        after_is_correct = (predictions == y_test)

        print("- Classification report:\n" + classification_report(y_test, predictions))

        # Plot scores differences only for the first fold as an example, only for synthetic dataset
        if simulation_parameters['dataset'] == 'synthetic' and i == 0:
            plot_scores(X_test, i, before_is_correct, after_is_correct, before_queries_score, performance_history[-1],
                        budget, simulation_parameters)

        supports_vector.append(probas)
        ground_truth_vector.append(y_test)

    print_mean_and_std(scores_vector)

    # Create the directory if it doesn't exist
    directory_path_titanic = 'saved_scores_titanic'
    directory_path_synthetic = 'saved_scores_synthetic'
    directory_path_roc_params_titanic = 'saved_roc_params_titanic'
    directory_path_roc_params_synthetic = 'saved_roc_params_synthetic'
    os.makedirs(directory_path_titanic, exist_ok=True)
    os.makedirs(directory_path_synthetic, exist_ok=True)
    os.makedirs(directory_path_roc_params_titanic, exist_ok=True)
    os.makedirs(directory_path_roc_params_synthetic, exist_ok=True)

    # Generate the file name
    scores_filename = f"scores_vector_for_{simulation_parameters['estimator']}_{simulation_parameters['dataset']}.npy"
    supports_filename = f"support_vector_for_{simulation_parameters['estimator']}_{simulation_parameters['dataset']}.npy"
    ground_truth_filename = f"ground_truth_vector_for_{simulation_parameters['estimator']}_{simulation_parameters['dataset']}.npy"

    if simulation_parameters['dataset'] == 'titanic':
        save_to_npy(directory_path_titanic, scores_filename, scores_vector)
        save_to_npy(directory_path_roc_params_titanic, supports_filename, supports_vector)
        save_to_npy(directory_path_roc_params_titanic, ground_truth_filename, ground_truth_vector)
    else:
        save_to_npy(directory_path_synthetic, scores_filename, scores_vector)
        save_to_npy(directory_path_roc_params_synthetic, supports_filename, supports_vector)
        save_to_npy(directory_path_roc_params_synthetic, ground_truth_filename, ground_truth_vector)

    plot_accuracy(accuracy_history, simulation_parameters)


def __get_n_neighbours(d, key, default=3):
    try:
        return int(d[key])
    except KeyError:
        return default
