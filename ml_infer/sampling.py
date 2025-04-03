from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter


def smote_oversample(windows_X, windows_y):
    num_features = windows_X.shape[1] * windows_X.shape[2]
    X_flat = windows_X.reshape(windows_X.shape[0], num_features)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_flat, windows_y)
    X_resampled = X_resampled.reshape(-1, windows_X.shape[1], windows_X.shape[2])
    return X_resampled, y_resampled


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
def over_under_sample_windows(windows_X, windows_y):
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

    # Oversample minority classes
    X_resampled, y_resampled = oversample_windows(X_under.reshape(-1, windows_X.shape[1], windows_X.shape[2]), y_under)
    return X_resampled, y_resampled


def over_under_sample(X, y):
    under_ratio = 0.5
    class_counts = Counter(y)
    min_class_count = min(class_counts.values())

    # Create dictionary for undersampling strategy
    under_strategy = {cls: int(count * under_ratio) if count * under_ratio > min_class_count else count
                      for cls, count in class_counts.items()}
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)
    X_under, y_under = rus.fit_resample(X, y)

    # Oversample
    ros = RandomOverSampler(random_state=69)
    X_resampled, y_resampled = ros.fit_resample(X_under, y_under)
    return X_resampled, y_resampled
