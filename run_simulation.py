import inquirer


estimators = [
    inquirer.List('estimator',
                  message="Choose estimator used for Active Learning simulation:",
                  choices=['GaussianNB', 'kNearestNeighbours', 'sth...'])
]

choices = inquirer.prompt(estimators)
print(choices)
