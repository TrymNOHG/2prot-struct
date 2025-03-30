from collections import defaultdict
import random

random.seed(42)

class StochasticModel:
    def __init__(self):
        self.probs = None

    def fit(self, y):
        self.probs = defaultdict(int)
        sum_vals = 0
        for input in y:
            sum_vals += len(input)
            for dssp in input:
                self.probs[dssp.upper()] += 1
        for keys in self.probs:
            self.probs[keys] /= sum_vals

    def predict(self, X):
        if self.probs is None:
            raise ValueError("Model must be fit before evaluation.")
        predictions = []
        for aa in X:
            pred = []
            for residue in aa:
                rand = random.random()
                proba = 0
                for dssp in self.probs:
                    proba += self.probs[dssp]
                    if rand < proba:
                        pred.append(dssp)
                        break
            predictions.append("".join(pred))
        return predictions


    def evaluate(self, X, y):
        if self.probs is None:
            raise ValueError("Model must be fit before evaluation.")
        predictions = self.predict(X)
        correct = 0
        total = 0
        i = 0
        for index, value in y.items():
            for (j, val) in enumerate(value):
                correct += 1 if predictions[i][j] == val else 0
                total += 1
            i += 1
        print(f"Accuracy of the model is {correct/total}")
        return correct/total
