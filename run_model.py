import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from sampling import over_under_sample_windows
from models.mlp_window_model import MLPModel
from models.stochastic_model import StochasticModel
from models.naive_bayes_model import NaiveBayesModel
from models.simple_window_model import TreeWindowModel
import eval_model
# from gen_graphs import *


def store_data(X_train, y_train, X_test, y_test, X_val, y_val, filename):
    with open(filename, 'wb') as f:
        pickle.dump((X_train, y_train, X_test, y_test, X_val, y_val), f, protocol=pickle.HIGHEST_PROTOCOL)


def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def store_predictions(y_test, y_pred, filename):
    with open(filename, 'wb') as f:
        pickle.dump((y_test, y_pred), f, protocol=pickle.HIGHEST_PROTOCOL)


LOAD_DATA = False
LOAD_DATA_FILENAME = "data/over_under_sampled_data.pkl"
STORE_DATA = False
STORE_DATA_FILENAME = "data/over_under_sampled_data.pkl"
STORE_PREDICTIONS = False
STORE_PREDICTIONS_FILENAME = "data/predictions.pkl"


# ['ABC', 'DEF'] -> ['A', 'B', 'C', 'D' *E, 'F']
def flatten_sequence(y):
    return [pred for seq in y for pred in seq]


# Load data
# Split data
# Sample?
# Train
# Predict on test set
# Evaluate
# TODO: Make generic later
def run_stochastic():
    model = StochasticModel()

    df = pd.read_csv("data/data.csv")
    y = df['dssp8'][:].values
    X = df['input'][:].values

    # Note: the validation nor test split keep the original dataset distribution balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    print("Number of data points in training set", len(X_train))
    _, _ = model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test = flatten_sequence(y_test)
    y_pred = flatten_sequence(y_pred)

    results = eval_model.evaluate_classification(y_test, y_pred)
    eval_model.evaluation_summary(results)


def run_naive_bayes():
    model = NaiveBayesModel()

    df = pd.read_csv("data/data.csv")
    y = df['dssp8']
    X = df['input']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(flatten_sequence(X_test))

    y_test = flatten_sequence(y_test)
    y_pred = flatten_sequence(y_pred)

    results = eval_model.evaluate_classification(y_test, y_pred)
    eval_model.evaluation_summary(results)


def run_tree_window():
    model = TreeWindowModel(window_length=5)
    df = pd.read_csv("data/data.csv")

    y = df['dssp8']
    X = df['input']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)

    y_pred = model.predict(flatten_sequence(X_test))
    y_test = flatten_sequence(y_test)

    results = eval_model.evaluate_classification(y_test, y_pred)
    eval_model.evaluation_summary(results)



if __name__ == "__main__":
    run_tree_window()
    # #model = MLPModel(window_length=17)
    # model = StochasticModel()

    # if LOAD_DATA:
    #     X_train, y_train, X_test, y_test, X_val, y_val = load_data(LOAD_DATA_FILENAME)
    # else:
    #     df = pd.read_csv("data/data.csv")
    #     y = df['dssp8'][:1_000].values
    #     X_original = df['input'][:1_000].values

    #     #X, y = model.to_windows(X_original, y)
    #     X = X_original

    #     # Note: the validation nor test split keep the original dataset distribution balance
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    #     # X_train, y_train = over_under_sample_windows(X_train, y_train)
    #     
    #     print("Number of data points in training set", len(X_train))
    #     if STORE_DATA:
    #         store_data(X_train, y_train, X_test, y_test, X_val, y_val, STORE_DATA_FILENAME)

    # # train_losses, validation_losses = model.fit(X_train, y_train, X_val, y_val)

    # train_losses, validation_losses = model.fit(X_train, y_train)

    # # loss_graph(train_losses, validation_losses)
    # y_pred = model.predict(X_test)

    # if STORE_PREDICTIONS:
    #     store_predictions(y_test, y_pred, STORE_PREDICTIONS_FILENAME)

    # results = eval_model.evaluate_classification(stochastic_pred_to_regular_pred(y_test), stochastic_pred_to_regular_pred(y_pred))
    # eval_model.evaluation_summary(results)


