import os
import pathlib
import sys

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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


def print_mean_and_std(before_queries_results, after_queries_results):
    table = [[round(np.average(before_queries_results), 3), round(np.average(after_queries_results), 3)],
             [round(np.std(before_queries_results), 3), round(np.std(after_queries_results), 3)]]

    df = pd.DataFrame(table,
                      columns=['without AL', 'with AL'],
                      index=['Mean', 'Std'])

    print('Mean and std for scores:\n', df.to_markdown())


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
