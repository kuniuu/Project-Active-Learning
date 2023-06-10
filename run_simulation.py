from matplotlib import pyplot as plt

from methods.active_learning.consts import ESTIMATORS
from methods.statistical_tests.t_test import t_test
from methods.utils import set_seed
from scripts.active_learning import run_active_learning

set_seed(2137)

for estimator in ESTIMATORS:
    run_active_learning(estimator)

t_test()
t_test(for_titanic=True)

plt.show()
