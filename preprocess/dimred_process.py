from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle as pk
import gc

def split_by_indices(X, y, save_file=None):
    indices = range(len(X))

    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=42)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    split_type = []

    for _ in train_indices:
        split_type.append('train')
    for _ in val_indices:
        split_type.append('val')
    for _ in test_indices:
        split_type.append('test')

    df = pd.DataFrame(data=[np.concatenate((train_indices, val_indices, test_indices)), split_type], columns=['indices', 'split'])
    if save_file is not None:
        df.to_csv(save_file if '.png' in save_file else f"{save_file}.png")
            
    return X_train, y_train, X_test, y_test, X_val, y_val

def split_by_val(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_test, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    new_X_df = pd.concat([X_train, X_val, X_test])
    new_y = []
    new_y.extend(y_train) 
    new_y.extend(y_val) 
    new_y.extend(y_test) 

    new_X_df['dssp8'] = new_y

    train = ['train' for _ in range(len(X_train))]
    val = ['val' for _ in range(len(X_val))]
    test = ['test' for _ in range(len(X_test))]
    split = []
    split.extend(train)
    split.extend(val)
    split.extend(test)

    new_X_df['split'] = split

    new_X_df.to_csv("split.csv")
    return X_train, y_train, X_test, y_test, X_val, y_val

data = pd.read_csv("./pLM/embeddings.csv")

y = data[['aa', 'dssp8']]
X = data.drop(columns=['aa', 'dssp8'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

pca = PCA(0.95) # I want to retain 95% variance.
pca.fit(X_train, y_train)

pk.dump(pca,  open("./dimred_data/95_pca.pkl","wb"))

X_train_vals = pca.transform(X_train)
X_val_vals = pca.transform(X_val)
X_test_vals = pca.transform(X_test)

print(len(X_train_vals[0]))
print(len(X_val_vals[0]))
print(len(X_test_vals[0]))

dimred_X = np.vstack((X_train_vals, X_val_vals, X_test_vals))
#dimred_X = np.concatenate((X_train_vals, X_val_vals), axis=0)
#dimred_X = np.concatenate((dimred_X, X_test_vals), axis=0)



y = pd.concat([y_train, y_val], ignore_index=True)
y = pd.concat([y, y_test], ignore_index=True)

split = []
split.extend(['train' for i in range(len(y_train['dssp8']))])
split.extend(['val' for i in range(len(y_val['dssp8']))])
split.extend(['test' for i in range(len(y_test['dssp8']))])

df = pd.DataFrame(data=dimred_X, columns=[i for i in range(len(X_train_vals[0]))])
df = pd.concat([df, y.reset_index(drop=True)], axis=1)
df['split'] = split

df.to_csv("./dimred_data/95_var_dimred_embedded.csv")