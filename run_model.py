import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/data.csv")

y = df['dssp8']
X_original = df['input']

from models.mlp_window_model import MLPWindowModel

model = MLPWindowModel(window_length=17)
X, y = model.to_windows(X_original, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
model.evaluate(X_test, y_test)
"""
X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42)

from models.simple_window_model import TreeWindowModel

#model = TreeWindowModel(window_length=5)
#model.fit(X_train, y_train)
#print(model.evaluate(X_test, y_test))

model = MLPWindowModel(window_length=window_length)
model.fit(X_train, y_train, X_test, y_test)
"""
