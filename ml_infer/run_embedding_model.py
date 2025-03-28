import json
import pandas as pd
import eval_model
import numpy as np
from sklearn.model_selection import train_test_split
from sampling import over_under_sample
import sys
print(sys.path)

import run_model
import gen_graphs
import gc

from models.mlp_window_model import MLPModel, MLPEmbeddings


# NOTE: A lot of code here is basically the same as run_model.py
#       These two files can be unified at some point, but for current experimentation it's eadsier to keep them separate

LOAD_DATA = False 
LOAD_DATA_FILENAME = "data/95_embedding_data.pkl"
STORE_DATA = True
STORE_DATA_FILENAME = "data/95_embedding_data.pkl"
STORE_PREDICTIONS = True 
STORE_PREDICTIONS_FILENAME = "data/95_predictions.pkl"

if __name__ == "__main__":
    model = MLPModel(model=MLPEmbeddings())

    if LOAD_DATA:
        X_train, y_train, X_test, y_test, X_val, y_val = run_model.load_data(LOAD_DATA_FILENAME)
    else:
        df = pd.read_csv("../../2prot-struct/dimred_data/95_var_dimred_embedded.csv")
        # X = df["input"].map(json.loads)
        y = df["dssp8"].values
        secondary_structures = "HECTGSPIB"
        secondary_structure_to_idx = {ss: i for i, ss in enumerate(secondary_structures)}
        y = np.array([secondary_structure_to_idx[label] for label in y])
        split = df["split"].values
        X = df.drop(columns=['aa', 'dssp8', 'split']).values
        del df 
        gc.collect()
        X_train = X[split == "train"]
        y_train = y[split == "train"]
        X_val = X[split == "val"]
        y_val = y[split == "val"]
        X_test = X[split == "test"]
        y_test = y[split == "test"]
        del X
        del y
        gc.collect()
        print("Number of data points in training set", len(X_train))
        if STORE_DATA:
            run_model.store_data(X_train, y_train, X_test, y_test, X_val, y_val, STORE_DATA_FILENAME)

    train_losses, validation_losses = model.fit(X_train, y_train, X_val, y_val, epochs=5)
    gen_graphs.loss_graph(train_losses, validation_losses)


    y_pred = model.predict(X_test)

    if STORE_PREDICTIONS:
        run_model.store_predictions(y_test, y_pred, STORE_PREDICTIONS_FILENAME)

    results = eval_model.evaluate_classification(y_test, y_pred)
    eval_model.evaluation_summary(results)
    results = eval_model.evaluate_classification(y_test, y_pred)
    eval_model.evaluation_summary(results)