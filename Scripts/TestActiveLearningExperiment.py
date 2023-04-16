import numpy
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.naive_bayes import GaussianNB

# Define a synthetic dataset of 400 samples with 2 informative features and 2 classes (binary problem)
X_raw, y_raw = make_classification(
    n_samples=400,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    flip_y=0.08
)

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

# Create a RepeatedStratifiedKFold loop
for i, (train_index, test_index) in enumerate(rskf.split(X_raw, y_raw)):
    # Define train and test sets
    X_train, X_test = X_raw[train_index], X_raw[test_index]
    y_train, y_test = y_raw[train_index], y_raw[test_index]

    # Find 5 random indexes for new ground true train subset
    n_labeled_examples = X_train.shape[0]
    training_indices = numpy.random.randint(low=0, high=n_labeled_examples, size=5)

    # From train set, create a new ground true train subset
    X_train_new = X_train[training_indices]
    y_train_new = y_train[training_indices]

    # From train set, create a new pooling subset
    X_pool = numpy.delete(X_train, training_indices, axis=0)
    y_pool = numpy.delete(y_train, training_indices, axis=0)

    # Learner initialization with ground true train subset, Gaussian Naive-Bayes estimator and Least of Confidence sampling method
    learner = ActiveLearner(
        estimator=GaussianNB(),
        query_strategy=uncertainty_sampling,
        X_training=X_train_new,
        y_training=y_train_new
    )

    # Make a prediction and check if it was correct (for scatter purposes only)
    predictions = learner.predict(X_test)
    is_correct = (predictions == y_test)

    # Calculate and report our model's accuracy, then append it to the noqueried_vector
    unqueried_score = learner.score(X_test, y_test)
    print('\nAccuracy before queries in repeat {i}: {acc:0.4f}'.format(i=i + 1, acc=unqueried_score))
    noqueried_vector.append(unqueried_score)

    # Scatter no-pooling classification
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    ax.scatter(X_test[:, 0][is_correct], X_test[:, 1][is_correct], c='g', marker='+', label='Correct', alpha=8/10)
    ax.scatter(X_test[:, 0][~is_correct], X_test[:, 1][~is_correct], c='r', marker='x', label='Incorrect', alpha=8/10)
    ax.legend(loc='lower right')
    ax.set_title('ActiveLearner class prediction without pooling in repeat {i} (Accuracy: {score:.3f})'.format(
        i=i + 1, score=unqueried_score))
    plt.show()

    # Set up a budget (30% of pooling subset)
    budget = round(0.3 * X_pool.shape[0])
    print('Budget: ', budget)
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

        # Calculate and report our model's accuracy.
        queried_score = learner.score(X_test, y_test)
        print('Accuracy after query {n} in repeat {i}: {acc:0.4f}'.format(n=index + 1, i=i + 1, acc=queried_score))

        # Save our model's performance for plotting.
        performance_history.append(queried_score)

    # Append last model's performance instance to the queried_vector
    queried_vector.append(performance_history[-1])

    # Make a prediction and check if it was correct (for scatter purposes only)
    predictions = learner.predict(X_test)
    is_correct = (predictions == y_test)

    # Scatter after-pooling classification
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=130)
    ax.scatter(X_test[:, 0][is_correct], X_test[:, 1][is_correct], c='g', marker='+', label='Correct', alpha=8 / 10)
    ax.scatter(X_test[:, 0][~is_correct], X_test[:, 1][~is_correct], c='r', marker='x', label='Incorrect', alpha=8 / 10)
    ax.legend(loc='lower right')
    ax.set_title('ActiveLearner class prediction after {n} queries in repeat {i} (Accuracy: {final_acc:.3f})'.format(
        n=budget, i=i + 1, final_acc=performance_history[-1]))
    plt.show()

# Print mean and std of no-pooling accuracy scores
print("\nUnqueried vector data")
print("Mean ", round(numpy.average(noqueried_vector), 3))
print("Std ", round(numpy.std(noqueried_vector), 3))

# Print mean and std of after-pooling accuracy scores
print("\nQueried vector data")
print("Mean ", round(numpy.average(queried_vector), 3))
print("Std ", round(numpy.std(queried_vector), 3))