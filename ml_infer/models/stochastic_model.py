from collections import defaultdict
import random
import pickle

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
        for i in range(len(y)):
            for j in range(len(y[i])):
                correct += 1 if predictions[i][j] == y[i][j] else 0
                total += 1
        
        print(f"Accuracy of the model is {correct/total}")
        return correct/total

    def forward(self, X):
        if self.probs is None:
            raise ValueError("Model must be fit before evaluation.")
        
        # NOTE: The distributions are static, so maybe we can just return the learned prior ?
        output_distributions = []
        for aa in X:
            sequence_probs = []
            for _ in aa:
                # Each residue gets the same distribution from the learned prior
                sequence_probs.append(dict(self.probs))
            output_distributions.append(sequence_probs)
        return output_distributions

    def pickle_model(self):
        file_name = "pickled_models/stochastic_model.pkl"
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)