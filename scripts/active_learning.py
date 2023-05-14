import numpy

from matplotlib import pyplot as plt

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB

from modAL.models import ActiveLearner

# Our own least_confidence method
from methods.my_least_confidence import least_confidence_sampling
from methods.utilities import plot_scores, plot_accuracy
from datasets.artificial_dataset import create_dataset

# Define random state
RANDOM_STATE = 2137

# Define a synthetic dataset
X_raw, y_raw = create_dataset(RANDOM_STATE)

# Plot raw dataset
plt.figure(figsize=(8.5, 6), dpi=130)
plt.scatter(X_raw[:, 0], X_raw[:, 1], c=y_raw)
plt.title('Raw dataset')
plt.show()

# Define RepeatedStratifiedKFold with 2 splits and 5 repeats
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

# Define vectors for accuracy scores - noqueried (no pooling) and queried (pooling)
noqueried_vector = []
queried_vector = []
# Define a list for accuracy history of each fold
accuracy_history = []

# Create a RepeatedStratifiedKFold loop
for i, (train_index, test_index) in enumerate(rskf.split(X_raw, y_raw)):
    # Define train and test sets
    X_train, X_test = X_raw[train_index], X_raw[test_index]
    y_train, y_test = y_raw[train_index], y_raw[test_index]

    # Find 5 random indexes for new ground true train subset
    n_labeled_examples = X_train.shape[0]
    training_indices = numpy.random.randint(low=0, high=n_labeled_examples, size=50)

    # From train set, create a new ground true train subset
    X_train_new = X_train[training_indices]
    y_train_new = y_train[training_indices]

    # From train set, create a new pooling subset
    X_pool = numpy.delete(X_train, training_indices, axis=0)
    y_pool = numpy.delete(y_train, training_indices, axis=0)

    # Learner initialization with ground true train subset, Gaussian Naive-Bayes estimator and Least of Confidence sampling method
    learner = ActiveLearner(
        estimator=GaussianNB(),
        query_strategy=least_confidence_sampling,
        X_training=X_train_new,
        y_training=y_train_new
    )

    # Make a prediction and check if it was correct (for scatter purposes only)
    predictions = learner.predict(X_test)
    before_is_correct = (predictions == y_test)

    # Calculate and report our model's accuracy, then append it to the noqueried_vector
    unqueried_score = accuracy_score(y_test, predictions)
    print('\nAccuracy before queries in repeat {i}: {acc:0.4f}'.format(i=i + 1, acc=unqueried_score))
    noqueried_vector.append(unqueried_score)

    # Set up a budget (30% of pooling subset)
    budget = round(0.3 * X_pool.shape[0])
    # print('Budget: ', budget)
    performance_history = [unqueried_score]



    # Create a ActiveLearning loop
    for index in range(budget):
        # Query a pooling subset
        query_index, query_instance = learner.query(X_pool)

        # Teach our ActiveLearner model the record it has requested.
        X, y = X_pool[query_index], y_pool[query_index]
        learner.teach(X=X, y=y)

        # Remove the queried instance from the unlabeled pooling subset.
        X_pool, y_pool = numpy.delete(X_pool, query_index, axis=0), numpy.delete(y_pool, query_index)

        # Predict and calculate model's accuracy.
        queried_score = learner.score(X_test, y_test)

        # Save our model's performance for plotting.
        performance_history.append(queried_score)


    accuracy_history.append(performance_history)


    print('Accuracy after query {n} in repeat {i}: {acc:0.4f}'.format(n=budget, i=i + 1, acc=performance_history[-1]))

    # Append last model's performance instance to the queried_vector
    queried_vector.append(performance_history[-1])

    # Make a prediction and check if it was correct (for scatter purposes only)
    predictions = learner.predict(X_test)
    after_is_correct = (predictions == y_test)

    # Scatter after-pooling classification
    plot_scores(X_test, i, before_is_correct, after_is_correct, unqueried_score, performance_history[-1], budget)


# Print mean and std of no-pooling accuracy scores
plt.show()

print("\nUnqueried vector data")
print("Mean ", round(numpy.average(noqueried_vector), 3))
print("Std ", round(numpy.std(noqueried_vector), 3))

# Print mean and std of after-pooling accuracy scores
print("\nQueried vector data")
print("Mean ", round(numpy.average(queried_vector), 3))
print("Std ", round(numpy.std(queried_vector), 3))

# Chart for accuracy growth
plot_accuracy(accuracy_history)
plt.show()