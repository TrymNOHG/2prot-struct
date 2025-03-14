import os
import matplotlib.pyplot as plt
from run_model import load_data, LOAD_DATA_FILENAME


def data_distribution():
    secondary_structures = "HECTGSPIB"
    secondary_structure_to_idx = {ss: i for i, ss in enumerate(secondary_structures)}
    idx_to_secondary_structure = {i: ss for ss, i in secondary_structure_to_idx.items()}

    # Load data, assumes the data is already stored at some other point.
    _, y_train, _, y_test, _, y_val = load_data(LOAD_DATA_FILENAME)

    # Convert back to secondary structure characters
    y_train_labels = [idx_to_secondary_structure[idx] for idx in y_train]
    y_test_labels = [idx_to_secondary_structure[idx] for idx in y_test]
    y_val_labels = [idx_to_secondary_structure[idx] for idx in y_val]

    # Count occurrences of each secondary structure
    def count_labels(y_labels):
        return {ss: y_labels.count(ss) for ss in secondary_structures}

    train_counts = count_labels(y_train_labels)
    test_counts = count_labels(y_test_labels)
    val_counts = count_labels(y_val_labels)

    # Ensure graphs directory exists
    os.makedirs("graphs", exist_ok=True)

    # Plot distributions and save them
    def plot_distribution(counts, title, filename):
        plt.figure(figsize=(8, 5))
        plt.bar(counts.keys(), counts.values(), alpha=0.7)
        plt.xlabel("Secondary Structure")
        plt.ylabel("Count")
        plt.title(title)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(os.path.join("graphs", filename))
        plt.close()

    plot_distribution(train_counts, "Train Set Distribution", "train_distribution.png")
    plot_distribution(test_counts, "Test Set Distribution", "test_distribution.png")
    plot_distribution(val_counts, "Validation Set Distribution", "val_distribution.png")


def loss_graph(train_losses, validation_losses):
    os.makedirs("graphs", exist_ok=True)

    # Start epochs from 1
    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Training Loss", marker="o", linestyle="-")
    plt.plot(epochs, validation_losses, label="Validation Loss", marker="s", linestyle="--")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # Save the figure
    plt.savefig(os.path.join("graphs", "loss_curve.png"))
    plt.close()


if __name__ == "__main__":
    data_distribution()
