import numpy as np
from scipy.stats import ttest_rel


def t_test(before_queries_results, after_queries_results):
    alpha = 0.05

    t_stats = np.zeros(1)
    p_values = np.zeros(1)
    better_scores = np.zeros(1, dtype=bool)

    # t, p = ttest_rel(before_queries_results, after_queries_results)
    t, p = ttest_rel(after_queries_results, before_queries_results)
    t_stats[0] = t
    p_values[0] = p
    better_scores[0] = np.mean(after_queries_results) > np.mean(before_queries_results)

    stat_adv = p_values < alpha
    significant_stat_adv = stat_adv * better_scores

    print('\n-----------------------------------------------------------------')
    print('t-test results:\n')
    print('- The t-statistic: {t_stats}'.format(t_stats=t_stats))
    print('- The p-value: {p_values}'.format(p_values=p_values))
    print('- After-queries results are better: {better_scores}'.format(better_scores=better_scores))
    print('- It is statistical advantage: {stat_adv}'.format(stat_adv=stat_adv))
    print('- It is statistically significant advantage: {significant_stat_adv}'.format(significant_stat_adv=significant_stat_adv))
    print('-----------------------------------------------------------------\n')
