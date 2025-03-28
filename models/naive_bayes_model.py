import numpy as np
import random
from collections import defaultdict


# Simply calculate the relative probability of structure given a certain amino acid
class NaiveBayesModel:
    def __init__(self):
        # probs essentially represents p(lab | aa)?
        self.probs = {} # Key will represent amino acid. Value will represent probability for label.
        self.classes = ['G', 'H', 'I', 'E', 'B', 'T', 'S', 'C', 'P']
        self.aa = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.p_aa = defaultdict(int)
        self.p_struct = defaultdict(int)

    def fit(self, X, y):
        for (i, seq) in enumerate(X.iloc):
            lab = y.iloc[i]
            for j in range(len(seq)):
                structure_index = self.classes.index(lab[j].upper())
                aa = seq[j].upper()
                self.p_aa[aa] += 1
                self.p_struct[lab[j].upper()] += 1
                if self.probs.get(aa) is None:
                    self.probs[aa] = [0 for _ in range(len(self.classes))]
                self.probs[aa][structure_index] += 1

        for key in self.probs:
            total = sum(self.probs[key])
            for i in range(len(self.probs[key])):
                self.probs[key][i] /= total

        total = sum(self.p_aa.values())
        for key in self.p_aa:
            self.p_aa[key] /= total

        total = sum(self.p_struct.values())
        for key in self.p_struct:
            self.p_struct[key] /= total

    def predict(self, X: list):
        vals = []
        for x in X:
            probs = np.array(self.probs[x.upper()])
            vals.append(self.classes[np.argmax(probs)])
        return "".join(vals)
    
    def predict_rand(self, X: list):
        output = []
        for x in X:
            probs = np.array(self.probs[x.upper()])
            rand = random.random()
            i = 0 
            while rand > probs[i]:
                rand -= probs[i]
                i += 1
            output.append(self.classes[i])
        return "".join(output)
    
    def predict_bayes(self, X):
        output = []
        for x in X:
            probs = np.array([])
            for struct in self.classes:
                probability = self.probs[x.upper()][self.classes.index(struct)] * self.p_struct[struct] / self.p_aa[x.upper()]
                # probability = self.probs[x.upper()][self.classes.index(struct)] * self.p_struct[struct]
                probs = np.append(probs, probability)
            output.append(self.classes[np.argmax(probs)])
        return "".join(output)
    
    def predict_bayes_rand(self, X):
        output = []
        for x in X:
            probs = np.array([])
            for struct in self.classes:
                probability = self.probs[x.upper()][self.classes.index(struct)] * self.p_struct[struct] / self.p_aa[x.upper()]
                # probability = self.probs[x.upper()][self.classes.index(struct)] * self.p_struct[struct]
                probs = np.append(probs, probability)
            rand = random.random()
            i = 0 
            while rand > probs[i]:
                rand -= probs[i]
                i += 1
            output.append(self.classes[i])
        return "".join(output)



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
        return np.mean(accuracies)

    def evaluate_rand(self, X, y):
        accuracies = []
        for i in range(len(X)):
            prediction = self.predict_rand(X.iloc[i])
            actual_pred = y.iloc[i]
            correct = 0
            for j in range(len(prediction)):
                correct += 1 if prediction[j] == actual_pred[j].upper() else 0
            accuracy = correct / len(actual_pred)
            accuracies.append(accuracy)
        return np.mean(accuracies)
    
    def evaluate_bayes(self, X, y):
        accuracies = []
        for i in range(len(X)):
            prediction = self.predict_bayes(X.iloc[i])
            actual_pred = y.iloc[i]
            correct = 0
            for j in range(len(prediction)):
                correct += 1 if prediction[j] == actual_pred[j].upper() else 0
            accuracy = correct / len(actual_pred)
            accuracies.append(accuracy)
        return np.mean(accuracies)
    
    def evaluate_bayes_rand(self, X, y):
        accuracies = []
        for i in range(len(X)):
            prediction = self.predict_bayes_rand(X.iloc[i])
            actual_pred = y.iloc[i]
            correct = 0
            for j in range(len(prediction)):
                correct += 1 if prediction[j] == actual_pred[j].upper() else 0
            accuracy = correct / len(actual_pred)
            accuracies.append(accuracy)
        return np.mean(accuracies)


# p(lab|aa) = p(aa|lab) * p(lab) / p(aa)