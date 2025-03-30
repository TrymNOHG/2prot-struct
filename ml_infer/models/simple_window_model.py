from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# Accuracy on 2000 samples :: 0.3698978429666714
class TreeWindowModel:
    def __init__(self, window_length):
        self.WINDOW_LENGTH = window_length
        # self.model = DecisionTreeClassifier() # RandomForestClassifier
        self.model = RandomForestClassifier(max_depth=20)
        self.classes = ['G', 'H', 'I', 'E', 'B', 'T', 'S', 'C', 'P']

    
    def gen_train(self, X_train, y_train):
        X_new = pd.DataFrame(columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        y_new = pd.DataFrame(columns=self.classes)
        # for (i, seq) in enumerate(X_train.iloc[:10]): # If I want to restrict number of training data
        for (i, seq) in enumerate(X_train.iloc):
            lab = y_train.iloc[i]
            for j in range(len(seq)-self.WINDOW_LENGTH):
                X_new.loc[len(X_new)] = 0        
                y_new.loc[len(y_new)] = 0
                for k in range(self.WINDOW_LENGTH):
                    X_new.loc[len(X_new) - 1, seq[j+k].upper()] += 1
                    y_new.loc[len(y_new) - 1, lab[j+k].upper()] += 1
        return X_new, y_new


    def fit(self, X, y):
        X, y = self.gen_train(X, y)
        self.model.fit(X, y)

    def predict(self, input_seq):
        X_new = pd.DataFrame(columns=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        for i in range(len(input_seq)-self.WINDOW_LENGTH):
            X_new.loc[len(X_new)] = 0        
            for j in range(self.WINDOW_LENGTH):
                X_new.loc[len(X_new) - 1, input_seq[i+j].upper()] += 1
        y_pred = [np.array([0 for _ in range(len(self.classes))]) for _ in range(len(input_seq))]
        for i in range(len(X_new)):
            row = X_new.iloc[[i]]
            pred = np.array(self.model.predict(row)[0])
            for j in range(self.WINDOW_LENGTH):
                y_pred[i+j] += pred
        
        # Remove code below to retain the probability distribution of different classes
        for i, val in enumerate(y_pred):
            y_pred[i] = self.classes[np.argmax(val)]
        return "".join(y_pred)

    def evaluate(self, X, y):
        accuracies = []
        for i in range(len(X)):
            prediction = self.predict(X.iloc[i])
            actual_pred = y.iloc[i]
            correct = 0
            for j in range(len(prediction)):
                correct += 1 if prediction[j] == actual_pred[j].upper() else 0
            accuracy = correct / len(actual_pred)
            accuracies.append(accuracy)
        # print(accuracies)
        return np.mean(accuracies)



