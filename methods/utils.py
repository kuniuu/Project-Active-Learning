import os
import pathlib
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from tabulate import tabulate


def set_seed(seed):
    np.random.seed(seed)


def get_seed():
    return np.random.get_state()[1][0]


def plot_accuracy(accuracy_history, choices):
    fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = plt.subplots(2, 5, figsize=(60, 25), dpi=130)
    fig.subplots_adjust(wspace=0.4, hspace=0.4, top=0.85)
    fig.suptitle(f"Accuracy History\n{choices['estimator']}\n{choices['dataset']}")

    for i, arr in enumerate(accuracy_history):
        x = range(len(arr))
        y = arr
        ax = plt.subplot(2, 5, i + 1)
        __scatter_accuracy(x, y, ax)
        ax.set_title(f"Fold {i + 1}")


def generate_roc_curve(for_titanic=False):
    # Load the predictions and ground truth
    directory_path = 'saved_roc_params_titanic' if for_titanic else 'saved_roc_params_synthetic'
    result_list = os.listdir(directory_path)

    pre_ground_truth = [__load_result(directory_path, result) for result in result_list[:3]]
    pre_predictions_results = [__load_result(directory_path, result) for result in result_list[3:]]

    # Convert the lists to numpy arrays
    predictions_results = np.array(pre_predictions_results)
    ground_truth = np.array(pre_ground_truth)

    # Calculate the ROC curve and AUC for each estimator and fold
    all_fpr = np.zeros([3, 10, 3])
    all_tpr = np.zeros([3, 10, 3])
    all_auc = np.zeros([3, 10])

    for i in range(3):
        for j in range(10):
            fpr, tpr, thresholds = roc_curve(ground_truth[i, j, :], predictions_results[i, j, :])
            roc_auc = auc(fpr, tpr)
            all_fpr[i, j, :] = fpr
            all_tpr[i, j, :] = tpr
            all_auc[i, j] = roc_auc

    # Calculate mean and standard deviation of FPR, TPR, and AUC
    mean_fpr = np.mean(all_fpr, axis=1)
    mean_tpr = np.mean(all_tpr, axis=1)
    std_fpr = np.std(all_fpr, axis=1)
    std_tpr = np.std(all_tpr, axis=1)
    mean_auc = np.mean(all_auc, axis=1)

    estimators = ['DT', 'GNB', 'kNN']

    # Plot the mean ROC curve with shaded region between standard deviation range
    plt.figure()

    # Plot the ROC curve for each estimator
    for i in range(3):
        plt.plot(mean_fpr[i], mean_tpr[i], label=f'Mean ROC for {estimators[i]} (AUC = {mean_auc[i]:.2f})')
        plt.fill_between(mean_fpr[i], mean_tpr[i] - std_tpr[i], mean_tpr[i] + std_tpr[i], alpha=0.2)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if for_titanic:
        plt.title('ROC Curve for Titanic dataset')
    else:
        plt.title('ROC Curve for synthetic dataset')
    plt.legend(loc='lower right')


def print_mean_and_std(results):
    table = [[round(np.average(results), 3)],
             [round(np.std(results), 3)]]

    df = pd.DataFrame(table,
                      columns=['Value'],
                      index=['Mean', 'Std'])

    print('- Mean and std for scores:\n', df.to_markdown(), '\n')


def save_to_npy(directory_path, filename, vector):
    # Save the scores vector
    file_path = pathlib.Path(directory_path) / filename
    np.save(file_path, vector)

    # Validate if the path exists
    if os.path.exists(file_path):
        print(f"Directory '{directory_path}' and file '{filename}' were created successfully.")
    else:
        print(f"Failed to create the directory '{directory_path}' or save the file '{filename}'.")
        sys.exit(1)


def plot_scores(X_test, iteration, before_is_correct: bool, after_is_correct: bool, before_queries_score_source,
                after_queries_score_source, budget, simulation_parameters):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 10), dpi=130)

    __scatter_scores(X_test, before_is_correct, ax1)
    ax1.set_title(
        'Before pooling, {estimator} estimator, fold {i}, accuracy: {score:.3f}'
        .format(i=iteration + 1, score=before_queries_score_source, estimator=simulation_parameters['estimator']))

    __scatter_scores(X_test, after_is_correct, ax2)
    ax2.set_title(
        'After pooling ({n} queries), {estimator} estimator, fold {i}, accuracy: {final_acc:.3f}'
        .format(n=budget, i=iteration + 1, final_acc=after_queries_score_source, estimator=simulation_parameters['estimator']))


def plot_queried_pool(X_queried, y_queried, iteration, fold, simulation_parameters):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=130)

    __scatter_queries(X_queried, y_queried, iteration, ax)
    ax.set_title(
        'Queried pool, {estimator} estimator, {dataset} dataset, fold {fold}, {i} queries'
        .format(i=iteration + 1, estimator=simulation_parameters['estimator'], dataset=simulation_parameters['dataset'], fold=fold + 1))


def __scatter_scores(X_test, is_correct: bool, ax):
    ax.scatter(X_test[:, 0][is_correct], X_test[:, 1][is_correct], c='g', marker='+', label='Correct', alpha=8 / 10)
    ax.scatter(X_test[:, 0][~is_correct], X_test[:, 1][~is_correct], c='r', marker='x', label='Incorrect', alpha=8 / 10)
    ax.legend(loc='lower right')


def __scatter_queries(X_queried, y_queried, i, ax):
    ax.scatter(X_queried[:i + 1, 0][y_queried[:i + 1]], X_queried[:i + 1, 1][y_queried[:i + 1]],
               c='g', marker='o', label='Queried samples with 1 label', alpha=8 / 10)
    ax.scatter(X_queried[:i + 1, 0][~y_queried[:i + 1]], X_queried[:i + 1, 1][~y_queried[:i + 1]],
               c='r', marker='o', label='Queried samples with 0 label', alpha=8 / 10)
    ax.legend(loc='lower right')
    ax.set_xlim(-5, 5)  # Set x-axis limits to -5 and 5
    ax.set_ylim(-5, 5)  # Set y-axis limits to -5 and 5


def __scatter_accuracy(x, y, ax):
    ax.plot(x, y)
    ax.set_xlabel('Query')
    ax.set_ylabel('Accuracy')


def __load_result(directory_path, filename):
    path = os.path.join(directory_path, filename)
    data = np.load(path)
    return data.tolist()
