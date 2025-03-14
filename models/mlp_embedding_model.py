import random
import numpy as np
from tinygrad import Tensor, nn, Device, TinyJit

# print(Device.DEFAULT)
random.seed(42)


# NOTE: A lot of code here is basically the same as mlp_window_model.py
#       These two files can be unified at some point, but for current experimentation it's eadsier to keep them separate

class MLP:
    def __init__(self):
        self.l1 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 8)

    def __call__(self, x:Tensor) -> Tensor:
        x = self.l1(x.flatten(1)).relu()
        x = self.l2(x).relu()
        return self.l3(x)


class MLPEmbeddingModel:
    def __init__(self):
        self.model = MLP()
        self.optim = nn.optim.Adam(nn.state.get_parameters(self.model), lr=0.001)

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=5, batch_size=8):
        n_samples = X_train.shape[0]
        steps_per_epoch = n_samples // batch_size
        
        @TinyJit
        def training_step(X_batch, y_batch):
            Tensor.training = True  # Enable dropout
            self.optim.zero_grad()
            loss = self.model(X_batch).softmax().sparse_categorical_crossentropy(y_batch).backward()
            self.optim.step()
            return loss

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            epoch_loss = 0
            # Iterate through mini-batches
            for step in range(steps_per_epoch):
                start_idx = step * batch_size
                end_idx = min((step + 1) * batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                X_batch = Tensor(X_train[batch_indices])
                y_batch = Tensor(y_train[batch_indices])
                loss = training_step(X_batch, y_batch)

                epoch_loss += loss.item()
                
                if step % (steps_per_epoch // 10) == 0 and step > 0:
                    print(f"Epoch {epoch+1}/{epochs} - Step {step}/{steps_per_epoch} - loss: {loss.item():.4f}")
            
            avg_loss = epoch_loss / steps_per_epoch
            train_accuracy = self.evaluate(X_train, y_train)
            if X_train is not None and y_train is not None:
                test_accuracy = self.evaluate(X_val, y_val)
                print(f"Epoch {epoch+1}/{epochs} - avg_loss: {avg_loss:.4f} - train accuracy: {train_accuracy:.2f} - test accuracy: {test_accuracy:.2f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - avg_loss: {avg_loss:.4f} - train accuracy: {train_accuracy:.2f}")

    def predict(self, X):
        @TinyJit
        def predict(X):
            Tensor.training = False
            return self.model(X).softmax().argmax(axis=1)
        
        predictions = predict(Tensor(X)).numpy()
        return predictions

    def evaluate(self, X, y):
        # If not already, turn into TinyGrad tensors
        if not isinstance(X, Tensor):
            X = Tensor(X)
        if not isinstance(y, Tensor):
            y = Tensor(y)

        @TinyJit
        def evaluate(X, y):
            Tensor.training = False
            predictions = self.model(X).softmax().argmax(axis=1)
            accuracy = (predictions == y).mean()
            return accuracy

        accuracy = evaluate(X, y).item()
        # print(f"Accuracy on {X.shape[0]} windows :: {accuracy:.4f}%")
        return accuracy
