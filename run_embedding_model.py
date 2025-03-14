import json
import pandas as pd
import eval_model
import numpy as np
from sklearn.model_selection import train_test_split
from sampling import over_under_sample
from run_model import load_data, store_data

from models.mlp_embedding_model import MLPEmbeddingModel


# NOTE: A lot of code here is basically the same as run_model.py
#       These two files can be unified at some point, but for current experimentation it's eadsier to keep them separate

LOAD_DATA = False
LOAD_DATA_FILENAME = "data/embedding_data.pkl"
STORE_DATA = True
STORE_DATA_FILENAME = "data/embedding_data.pkl"
STORE_PREDICTIONS = False
STORE_PREDICTIONS_FILENAME = "data/predictions.pkl"

if __name__ == "__main__":
    model = MLPEmbeddingModel()

    if LOAD_DATA:
        X_train, y_train, X_test, y_test, X_val, y_val = load_data(LOAD_DATA_FILENAME)
    else:
        df = pd.read_csv("data/embeddings.csv")

        X = df["input"].map(json.loads)
        y = df["dssp8"].values  

        X = np.array(X[0][:200])
        y = np.array(list(y[0][:200]))

        secondary_structures = "HECTGSPIB"
        secondary_structure_to_idx = {ss: i for i, ss in enumerate(secondary_structures)}
        y = np.array([secondary_structure_to_idx[label] for label in y])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        X_train, y_train = over_under_sample(X_train, y_train)
        print("Number of windows in training set", len(X_train))
        if STORE_DATA:
            store_data(X_train, y_train, X_test, y_test, X_val, y_val, STORE_DATA_FILENAME)

    model.fit(X_train, y_train, X_val, y_val, epochs=5)
    y_pred = model.predict(X_test)

    # if STORE_PREDICTIONS:
    #     store_predictions(y_test, y_pred, STORE_PREDICTIONS_FILENAME)

    results = eval_model.evaluate_classification(y_test, y_pred)
    eval_model.evaluation_summary(results)
    results = eval_model.evaluate_classification(y_test, y_pred)
    eval_model.evaluation_summary(results)
