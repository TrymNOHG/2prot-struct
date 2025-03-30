import numpy as np
import random
import pickle
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

    def forward(self, X):
        if not self.probs:
            raise ValueError("Model must be fit before calling forward.")

        output_distributions = []
        for seq in X:
            sequence_probs = []
            for aa in seq:
                aa = aa.upper()
                if aa not in self.aa:
                    # If it's an unknown amino acid, assign uniform probabilities
                    prob_dist = {c: 1/len(self.classes) for c in self.classes}
                else:
                    # Use Bayes rule: P(structure | aa) ∝ P(aa | structure) * P(structure)
                    numerator = []
                    for i, c in enumerate(self.classes):
                        # P(aa | structure) = Bayes inversion of P(structure | aa), approx via:
                        # P(structure | aa) * P(aa) / P(structure)
                        # But since P(aa | structure) isn't directly stored, we approximate:
                        # P(structure | aa) * P(structure) / P(aa)
                        p_struct_given_aa = self.probs[aa][i]
                        p_struct = self.p_struct[c]
                        p_aa = self.p_aa[aa]
                        numerator.append((p_struct_given_aa * p_struct) / p_aa if p_aa > 0 else 0.0)

                    total = sum(numerator)
                    prob_dist = {
                        self.classes[i]: numerator[i] / total if total > 0 else 0.0
                        for i in range(len(self.classes))
                    }

                sequence_probs.append(prob_dist)
            output_distributions.append(sequence_probs)
    
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


    def pickle_model(self):
        file_name = "pickled_models/naive_bayes.pkl"
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)


# p(lab|aa) = p(aa|lab) * p(lab) / p(aa)