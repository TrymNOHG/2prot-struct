import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE

from models.mlp_window_model import MLPWindowModel
import eval_model

# NOTE CRO SMOOOOTE
def smote_oversample(windows_X, windows_y):
    num_features = windows_X.shape[1] * windows_X.shape[2]
    X_flat = windows_X.reshape(windows_X.shape[0], num_features)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_flat, windows_y)

    X_resampled = X_resampled.reshape(-1, windows_X.shape[1], windows_X.shape[2])
    return X_resampled, y_resampled


# NOTE SUPER AD HOC
def oversample_windows(windows_X, windows_y):
    # Flatten to 2D
    num_features = windows_X.shape[1] * windows_X.shape[2]
    X_flat = windows_X.reshape(windows_X.shape[0], num_features)

    ros = RandomOverSampler(random_state=69)
    X_resampled, y_resampled = ros.fit_resample(X_flat, windows_y)

    # Reshape back to original window shape
    X_resampled = X_resampled.reshape(-1, windows_X.shape[1], windows_X.shape[2])
    return X_resampled, y_resampled


df = pd.read_csv("data/data.csv")

y = df['dssp8'][:3_000]
X_original = df['input'][:3_000]

model = MLPWindowModel(window_length=17)
X, y = model.to_windows(X_original, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = smote_oversample(X_train, y_train)

model.fit(X_train, y_train, X_test, y_test, epochs=20)
y_pred = model.predict(X_test)

results = eval_model.evaluate_classification(y_test, y_pred)
eval_model.evaluation_summary(results)

"""
Trymmi stuff
X_train, X_test, y_train, y_test = train_test_split(X_original, y, test_size=0.2, random_state=42)

from models.simple_window_model import TreeWindowModel

#model = TreeWindowModel(window_length=5)
#model.fit(X_train, y_train)
#print(model.evaluate(X_test, y_test))

model = MLPWindowModel(window_length=window_length)
model.fit(X_train, y_train, X_test, y_test)
"""
