import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from sampling import over_under_sample_windows
from models.mlp_window_model import MLPModel
import eval_model
from gen_graphs import *


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


if __name__ == "__main__":
    model = MLPModel(window_length=17)

    if LOAD_DATA:
        X_train, y_train, X_test, y_test, X_val, y_val = load_data(LOAD_DATA_FILENAME)
    else:
        df = pd.read_csv("data/data.csv")
        y = df['dssp8'][:1_000]
        X_original = df['input'][:1_000]
        X, y = model.to_windows(X_original, y)

        # Note: the validation nor test split keep the original dataset distribution balance
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, y_train = over_under_sample_windows(X_train, y_train)
        print("Number of windows in training set", len(X_train))
        if STORE_DATA:
            store_data(X_train, y_train, X_test, y_test, X_val, y_val, STORE_DATA_FILENAME)

    train_losses, validation_losses = model.fit(X_train, y_train, X_val, y_val, epochs=5)
    loss_graph(train_losses, validation_losses)
    y_pred = model.predict(X_test)

    if STORE_PREDICTIONS:
        store_predictions(y_test, y_pred, STORE_PREDICTIONS_FILENAME)

    results = eval_model.evaluate_classification(y_test, y_pred)
    eval_model.evaluation_summary(results)
