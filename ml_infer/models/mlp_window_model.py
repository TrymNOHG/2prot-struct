import random
import numpy as np
from tinygrad import Tensor, nn, Device, TinyJit

print(Device.DEFAULT)
random.seed(42)


class MLP:
    def __init__(self, window_length=17):
        # The network architecture is based on the below link:
        #  https://se.mathworks.com/help/bioinfo/ug/predicting-protein-secondary-structure-using-a-neural-network.html
        # First layer will be a one-hot encoding of the 20 amino acids for each position in the window
        self.l1 = nn.Linear(window_length * 20, 512)
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 8)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x.flatten(1)).relu()
        x = self.l2(x).relu()
        return self.l3(x)


class MLPEmbeddings:
    def __init__(self):
        self.l1 = nn.Linear(642, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 8)

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x.flatten(1)).relu()
        x = self.l2(x).selu()
        return self.l3(x)


class MLPWindowModel:
    def __init__(self, window_length=17, model=None):
        if window_length % 2 == 0:
            print("WARNING: Window length should probably be an odd number!")
        self.window_length = window_length
        if model is None:
            self.model = MLP(window_length=window_length)
        else:
            self.model = model
        self.optim = nn.optim.Adam(nn.state.get_parameters(self.model), lr=0.001)

    def to_windows(self, X, y):
        # secondary_structures = "HECTGSPIB"
        secondary_structures = ['G', 'H', 'I', 'E', 'B', 'T', 'S', 'C', 'P']
        secondary_structure_to_idx = {ss: i for i, ss in enumerate(secondary_structures)}
        # print(secondary_structure_to_idx)

        amino_acids = "ACDEFGHIKLMNPQRSTVWY"
        num_amino_acids = len(amino_acids)
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

        # Calculate total number of windows
        total_windows = sum(len(seq) - self.window_length + 1 for seq in X)
        # Pre-allocate arrays
        windows_X = np.zeros((total_windows, self.window_length, num_amino_acids), dtype=np.float32)
        windows_y = np.zeros(total_windows, dtype=np.int32)

        window_idx = 0
        for seq_idx in range(len(X)):
            seq = X[seq_idx]
            struct = y[seq_idx]
            seq_len = len(seq)
            for i in range(seq_len - self.window_length + 1):
                window = seq[i:i + self.window_length]
                # NOTE: Output structure is the middle element of the window, same as the matlab example.
                #       May want to consider more big brain approaches in the future.
                label = struct[i + self.window_length // 2]
                # print(label, secondary_structure_to_idx[label])
                # Create "one-hot encoding" for the current amino acids in the window
                for pos, aa in enumerate(window):
                    if aa in aa_to_idx:
                        windows_X[window_idx, pos, aa_to_idx[aa]] = 1.0

                windows_y[window_idx] = secondary_structure_to_idx[label]
                # l = secondary_structure_to_idx[label]
                # if l > 2:
                #     l = 2
                # windows_y[window_idx] = l
                window_idx += 1

        print("Number of sequences", len(X))
        print("Number of windows", len(windows_X))
        return windows_X, windows_y

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=5, batch_size=512):
        @TinyJit
        def training_step(X_batch, y_batch):
            Tensor.training = True  # Enable dropout
            self.optim.zero_grad()
            loss = self.model(X_batch).softmax().sparse_categorical_crossentropy(y_batch).backward()
            self.optim.step()
            return loss

        @TinyJit
        def compute_loss(X, y):
            self.optim.zero_grad()
            loss = self.model(X).softmax().sparse_categorical_crossentropy(y).backward()
            return loss

        # NOTE: Large batch size reduces the amount of memory transfers and kernel launches
        #       but is worse for regularization(?)
        #       Maybe we can do an entire epoch in one kernel launch?
        #       Results: Utilization barely improved for smaller batch sizes
        @TinyJit
        def training_epoch():
            Tensor.training = True  # Enable dropout
            losss = Tensor.ones(steps_per_epoch)
            for _ in range(steps_per_epoch):
                batch_indices = Tensor.randint(batch_size, high=X_train.shape[0])

                X_batch = X_train[batch_indices]
                y_batch = y_train[batch_indices]

                self.optim.zero_grad()
                self.model(X_batch).softmax().sparse_categorical_crossentropy(y_batch).backward()
                self.optim.step()
                # losss = training_step(X_batch, y_batch)

            return losss.mean()

        n_samples = X_train.shape[0]
        # steps_per_epoch = (n_samples // batch_size) + 1
        steps_per_epoch = n_samples // batch_size

        # X_train = Tensor(X_train)
        # y_train = Tensor(y_train)
        X_train_as_tensor = Tensor(X_train)
        y_train_as_tensor = Tensor(y_train)
        X_val = Tensor(X_val)
        y_val = Tensor(y_val)

        train_losses = [compute_loss(X_train_as_tensor, y_train_as_tensor).numpy()]
        validation_losses = [compute_loss(X_val, y_val).numpy()]

        for epoch in range(epochs):
            # avg_loss = training_epoch().numpy()

            # Shuffle indices for this epoch
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

                # batch_indices = Tensor.randint(batch_size, high=X_train.shape[0])

                epoch_loss += loss.item()

                if steps_per_epoch > 10 and step % (steps_per_epoch // 10) == 0 and step > 0:
                    print(f"Epoch {epoch + 1}/{epochs} - Step {step}/{steps_per_epoch} - loss: {loss.item():.4f}")

            avg_loss = epoch_loss / steps_per_epoch
            train_accuracy = self.evaluate(X_train, y_train)
            if X_val is not None and y_val is not None:
                # train_loss = compute_loss(X_train_as_tensor, y_train_as_tensor)
                # train_losses.append(train_loss)
                train_losses.append(avg_loss)
                validation_loss = compute_loss(X_val, y_val).numpy()
                validation_losses.append(validation_loss)

                test_accuracy = self.evaluate(X_val, y_val)
                print(
                    f"Epoch {epoch + 1}/{epochs} - avg_loss: {avg_loss:.4f} - train accuracy: {train_accuracy:.2f} - test accuracy: {test_accuracy:.2f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} - avg_loss: {avg_loss:.4f} - train accuracy: {train_accuracy:.2f}")

        return train_losses, validation_losses

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