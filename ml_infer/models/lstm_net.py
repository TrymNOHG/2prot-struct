import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMNet(nn.Module):
    def __init__(self, input_size, out_size):
        super(LSTMNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.LayerNorm(256)
        self.lstm1 = nn.LSTM(input_size=256, hidden_size=256, num_layers=1, batch_first=True)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.LayerNorm(128)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True)

        self.fc3 = nn.Linear(64, out_size)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm1(x)
        x = self.relu(self.bn2(self.fc2(x[:, -1, :])))
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x, _ = self.lstm2(x)
        x = self.fc3(x[:, -1, :])
        return F.log_softmax(x, dim=1)


    def train(self, device, train_loader, optimizer, epoch):
        self.train()
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                print(f'Train Epoch: {epoch + 1} \tLoss: {loss.item():.6f}')
        return running_loss / len(train_loader)


    def evaluate(self, device, data_loader, mode="Evaluation"):
        self.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = self(data)
                total_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        total_loss /= len(data_loader.dataset)
        accuracy = correct / len(data_loader.dataset)
        print(f'\n{mode} set: Average loss: {total_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} '
            f'({100. * accuracy:.0f}%)\n')
        return total_loss, accuracy
