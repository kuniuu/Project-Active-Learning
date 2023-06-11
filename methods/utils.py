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


def __scatter_accuracy(x, y, ax):
    ax.plot(x, y)
    ax.set_xlabel('Query')
    ax.set_ylabel('Accuracy')


def __load_result(directory_path, filename):
    path = os.path.join(directory_path, filename)
    data = np.load(path)
    return data.tolist()
