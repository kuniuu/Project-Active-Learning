import os

import numpy as np
from scipy.stats import ttest_rel


def t_test(for_titanic=False):

    # Load pre-results
    pre_results = []
    directory_path = 'saved_scores_titanic' if for_titanic else 'saved_scores_synthetic'
    results_list = os.listdir(directory_path)

    for result in results_list:
        file = np.load(directory_path + '/' + result)
        pre_results.append(file.tolist())

    # Transpose pre-results to get results - [estimator, scores] -> [scores, estimator]
    results = np.array(pre_results).T

    alpha = 0.05

    t_stats = np.zeros([3, 3])
    p_values = np.zeros([3, 3])
    better_scores = np.zeros([3, 3], dtype=bool)

    for i in range(3):
        for j in range(3):
            t, p = ttest_rel(results[:, i], results[:, j])
            t_stats[i, j] = t
            p_values[i, j] = p
            better_scores[i, j] = np.mean(results[:, i]) > np.mean(results[:, j])

    stat_adv = p_values < alpha
    significant_stat_adv = stat_adv * better_scores

    dataset = 'Titanic' if for_titanic else 'synthetic'

    print('\n-----------------------------------------------------------------')
    print(f't-test results for {dataset} dataset:\n')
    print(significant_stat_adv, '\n')

    if np.mean(results[:, 0]) > np.mean(results[:, 1]):
        print('DT with {a} is better than GNB with {b}'.format(a=round(np.mean(results[:, 0]), 3),
                                                                b=round(np.mean(results[:, 1]), 3)))
    else:
        print('GNB with {b} is better than DT with {a}'.format(a=round(np.mean(results[:, 0]), 3),
                                                                b=round(np.mean(results[:, 1]), 3)))
    if np.mean(results[:, 1]) > np.mean(results[:, 2]):
        print('GNB with {a} is better than kNN with {b}'.format(a=round(np.mean(results[:, 1]), 3),
                                                               b=round(np.mean(results[:, 2]), 3)))
    else:
        print('kNN with {b} is better than GNB with {a}'.format(a=round(np.mean(results[:, 1]), 3),
                                                               b=round(np.mean(results[:, 2]), 3)))
    if np.mean(results[:, 0]) > np.mean(results[:, 2]):
        print('DT with {a} is better than kNN with {b}'.format(a=round(np.mean(results[:, 0]), 3),
                                                               b=round(np.mean(results[:, 2]), 3)))
    else:
        print('kNN with {b} is better than DT with {a}'.format(a=round(np.mean(results[:, 0]), 3),
                                                               b=round(np.mean(results[:, 2]), 3)))
    print('-----------------------------------------------------------------')
