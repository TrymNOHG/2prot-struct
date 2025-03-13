import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pickle

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


# Severe imbalance in dataset means oversampling creates a lot of new data. Too much data for us to process!
# Strategy here is to somewhat undersample the majority classes and then overample the minority classes
# For the first 3_000 input sequences it produces 594_548 windows
def over_under_sample(windows_X, windows_y):
    num_features = windows_X.shape[1] * windows_X.shape[2]
    X_flat = windows_X.reshape(windows_X.shape[0], num_features)
    
    under_ratio = 0.5
    class_counts = Counter(windows_y)
    min_class_count = min(class_counts.values())
    
    # Create dictionary for undersampling strategy
    under_strategy = {cls: int(count * under_ratio) if count * under_ratio > min_class_count else count 
                       for cls, count in class_counts.items()}
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
    X_under, y_under = rus.fit_resample(X_flat, windows_y)

    # Undersample majority classes
    X_resampled, y_resampled = oversample_windows(X_under.reshape(-1, windows_X.shape[1], windows_X.shape[2]), y_under)
    return X_resampled, y_resampled


def store_data(X_train, y_train, X_test, y_test, filename):
    with open(filename, 'wb') as f:
        pickle.dump((X_train, y_train, X_test, y_test), f, protocol=pickle.HIGHEST_PROTOCOL)

def load_data(X_train, y_train, X_test, y_test, filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


df = pd.read_csv("data/data.csv")

y = df['dssp8'][:]
X_original = df['input'][:]

model = MLPWindowModel(window_length=17)
X, y = model.to_windows(X_original, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, y_train = over_under_sample(X_train, y_train)
# for the first 3000 datapoints,
#  oversampling gives 1_447_173 
#  over_under gives     723_582
#  no sampling          475_638
print("Number of windows in training set", len(X_train))

store_data(X_train, y_train, X_test, y_test, "data/over_under_sampled_data.pkl")


exit(0)
model.fit(X_train, y_train, X_test, y_test, epochs=5)
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
