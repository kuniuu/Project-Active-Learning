# Synthetic Data Generation and Active Learning

In this project, we will generate a synthetic dataset of 400 samples with two informative features and two classes (binary problem) using the `make_classification` function. Then we will split the dataset into training and test sets using `RepeatedStratifiedKFold` (2 folds, 5 repetitions).

For each iteration of `RepeatedStratifiedKFold`, we will:

- Randomly select a few labeled examples from the training set to create a new subset for training. The difference between the training set and the new subset will be the subset reserved for querying (Pool-Based Sampling).
- Create a learning model using GaussianNB estimator and uncertainty sampling method, which uses Least Confidence to train the model using the new subset.
- Calculate the model's accuracy on the test set and save the result as noqueried_score.
- Define the `budget` as 30% of the number of examples in the querying subset.

For each `budget` step, we will:

- Ask the learning model for the most uncertain example.
- Label the chosen example.
- Train the model with the labeled example.
- Remove the chosen example from the querying subset.
- Calculate the model's accuracy on the test set and save the result as queried_score.

Finally, we will calculate statistics for noqueried_vector and queried_vector:

- Calculate the mean and standard deviation for queried_vector and noqueried_vector, then display the statistics.
