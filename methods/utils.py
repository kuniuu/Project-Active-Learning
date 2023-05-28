from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate


def set_seed(seed):
    np.random.seed(seed)


def get_seed():
    return np.random.get_state()[1][0]


def plot_scores(X_test, iteration, before_is_correct: bool, after_is_correct: bool, before_queries_score_source,
                after_queries_score_source, budget):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 10), dpi=130)

    __scatter_scores(X_test, before_is_correct, ax1)
    ax1.set_title(
        'Before pooling, fold {i}, accuracy: {score:.3f}'
        .format(i=iteration + 1, score=before_queries_score_source))

    __scatter_scores(X_test, after_is_correct, ax2)
    ax2.set_title(
        'After pooling ({n} queries), fold {i}, accuracy: {final_acc:.3f}'
        .format(n=budget, i=iteration + 1, final_acc=after_queries_score_source))


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


def __scatter_scores(X_test, is_correct: bool, ax):
    ax.scatter(X_test[:, 0][is_correct], X_test[:, 1][is_correct], c='g', marker='+', label='Correct', alpha=8 / 10)
    ax.scatter(X_test[:, 0][~is_correct], X_test[:, 1][~is_correct], c='r', marker='x', label='Incorrect', alpha=8 / 10)
    ax.legend(loc='lower right')


def __scatter_accuracy(x, y, ax):
    ax.plot(x, y)
    ax.set_xlabel('Query')
    ax.set_ylabel('Accuracy')
