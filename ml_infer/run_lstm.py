from models.ltsm_net import LSTMNet
import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt

def plot_loss(loss, dataset_type: str):
    plt.plot(range(1, len(loss) + 1), loss, marker='x', label=f'{dataset_type} Loss')
    plt.xlabel('Epochs')
    plt.ylabel(f'{dataset_type} Set Loss')
    plt.title('Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'ltsm_{dataset_type.lower()}_loss.png')

def get_dataloaders(df: pd.DataFrame):
    df.drop('Unnamed: 0', axis=1, inplace=True)

    X = df.iloc[:, :-3].values
    y = df["dssp8"].values
    split = df["split"].values

    secondary_structures = "HECTGSIBP"
    mapping = {label: idx for idx, label in enumerate(secondary_structures)}
    y_numeric = np.array([mapping[label] for label in y])

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y_numeric, dtype=torch.long)

    train_indices = np.where(split == "train")[0]
    test_indices = np.where(split == "test")[0]
    validation_indices = np.where(split == "validate")[0]

    X_train, y_train = X_tensor[train_indices], y_tensor[train_indices]
    X_test, y_test = X_tensor[test_indices], y_tensor[test_indices]
    X_validation, y_validation = X_tensor[validation_indices], y_tensor[validation_indices]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    val_dataset = TensorDataset(X_validation, y_validation)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    df = pd.read_csv("nico_99.csv")
    train_loader, val_loader, test_loader = get_dataloaders(df)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMNet(input_size=X.shape[1], out_size=9).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []

    for epoch in range(15):
        train_loss = model.train(device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        val_loss, val_acc = model.evaluate(device, val_loader, mode="Validation")
        validation_losses.append(val_loss)
        validation_accuracies.append(val_acc)
        scheduler.step()

    test_loss, test_acc = model.evaluate(device, test_loader, mode="Test")

    plot_loss(train_losses, "Train")
    plot_loss(validation_losses, "Validation")