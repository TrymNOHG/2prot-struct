import pandas as pd

def extract_splits(csv_file, validation=True):
    df = pd.read_csv(csv_file)
    if validation is False:
        train = df[df['split'].isin(['train', 'val'])]
    else:
        train = df[df['split'] == 'train']
    val = df[df['split'] == 'val']
    test = df[df['split'] == 'test']

    y_train = train['dssp8']
    X_train = train.drop(columns=['aa', 'dssp8', 'split', 'Unnamed: 0'])
    y_val = val['dssp8']
    X_val = val.drop(columns=['aa', 'dssp8', 'split', 'Unnamed: 0'])
    y_test = test['dssp8']
    X_test = test.drop(columns=['aa', 'dssp8', 'split', 'Unnamed: 0'])

    return X_train, y_train, X_val, y_val, X_test, y_test
