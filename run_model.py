import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

from sampling import over_under_sample
from models.mlp_window_model import MLPWindowModel
import eval_model


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


model = MLPWindowModel(window_length=17)

if LOAD_DATA:
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(LOAD_DATA_FILENAME)
else:
    df = pd.read_csv("data/data.csv")
    y = df['dssp8'][:100]
    X_original = df['input'][:100]
    X, y = model.to_windows(X_original, y)

    # Note: the validation nor test split keep the original dataset distribution balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train, y_train = over_under_sample(X_train, y_train)
    print("Number of windows in training set", len(X_train))
    if STORE_DATA:
        store_data(X_train, y_train, X_test, y_test, X_val, y_val, STORE_DATA_FILENAME)

model.fit(X_train, y_train, X_val, y_val, epochs=5)
y_pred = model.predict(X_test)

if STORE_PREDICTIONS:
    store_predictions(y_test, y_pred, STORE_PREDICTIONS_FILENAME)

results = eval_model.evaluate_classification(y_test, y_pred)
eval_model.evaluation_summary(results)

"""
Trymmi stuff
X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42)

from models.simple_window_model import TreeWindowModel
from models.stochastic_model import StochasticModel


# model = TreeWindowModel(window_length=17)
model = StochasticModel()
# model.fit(X_train, y_train)
model.fit(y_train)

print(model.evaluate(X_test, y_test))
"""
