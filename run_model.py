import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv("data/data.csv")


y = df['dssp8'][:1000]
X_original = df['input'][:1000]
X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42)

from models.simple_window_model import TreeWindowModel
from models.stochastic_model import StochasticModel


# model = TreeWindowModel(window_length=17)
model = StochasticModel()
# model.fit(X_train, y_train)
model.fit(y_train)

print(model.evaluate(X_test, y_test))
