import re

import inquirer

from methods.utilities import set_seed
from scripts.active_learning import run_active_learning

questions = [
    inquirer.List('estimator',
                  message="Choose an estimator used for Active Learning simulation:",
                  choices=['GaussianNB', 'kNearestNeighbours', 'DecisionTreeClassifier']),
    inquirer.List('dataset',
                  message="Choose a dataset used for Active Learning simulation:",
                  choices=['Synthetic dataset', 'Titanic real dataset'])
]

choices = inquirer.prompt(questions)

if choices['estimator'] == 'kNearestNeighbours':
    questions = [
        inquirer.Text('n_neighbors',
                      message="Choose a number of neighbours between 2 and 50:",
                      validate=lambda _, x: re.match('^(?:[2-4]\\d|50|[2-9])$', x))
    ]
    choices['n_neighbors'] = inquirer.prompt(questions)['n_neighbors']

if choices['dataset'] == 'Synthetic dataset':
    questions = [
        inquirer.Confirm('plots',
                         message="Do you want to see plots for each fold?")
    ]
    choices['plots'] = inquirer.prompt(questions)['plots']

print("Running Active Learning simulation with the following choices:")
print("Choice of estimator: " + choices['estimator'])
print("Choice of dataset: " + choices['dataset'])
if choices['estimator'] == 'kNearestNeighbours':
    print("Choice of n_neighbors: " + choices['n_neighbors'])

set_seed(2137)

run_active_learning(choices)
