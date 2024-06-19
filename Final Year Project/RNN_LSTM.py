import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from Forex_Data import read_csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
torch.cuda.set_device(0)
torch.set_default_dtype(torch.float32)
device = torch.device("cuda")


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x.unsqueeze(1), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def backtest(dataFrame, predictions, actual):
    dataCopy = dataFrame.copy()
    dataCopy = pd.DataFrame(data=dataCopy, columns=['open', 'high', 'low', 'close'])
    dataCopy['close'] = predictions
    dataCopy[['open', 'high', 'low', 'close']] = scaler.inverse_transform(dataCopy[['open', 'high', 'low', 'close']])
    dataCopy['target'] = np.where(dataCopy['close'].shift(-1) > dataCopy['close'], 1, 0)
    total_trades = 0
    correct_trades = 0
    balance = 100.0

    for pred, actual in zip(dataCopy['target'], actual):
        trade_return = 10 if pred == actual else -10
        balance += trade_return
        total_trades += 1
        if (pred == actual):
            correct_trades += 1

    accuracy = correct_trades / total_trades

    print(f'Total Trades : {total_trades}, Correct Trades : {correct_trades}, Balance : {balance}')

    return accuracy


def data_preprocessing():
    data = read_csv().drop(0)
    data[['open', 'high', 'low', 'close']] = scaler.fit_transform(data[['open', 'high', 'low', 'close']])

    X = data[['open', 'high', 'low', 'close']].values
    y = data['target'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.from_numpy(X_train).type(torch.Tensor)
    X_test_tensor = torch.from_numpy(X_test).type(torch.Tensor)
    y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor)
    y_test_tensor = torch.from_numpy(y_test).type(torch.Tensor)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, X_train, X_test, y_train, y_test


def trainer():
    X_train_tensor, X_test_tensor, y_train_tensor, _, _, X_test, _, y_test = data_preprocessing()
    input_size = 4
    hidden_size = 42
    num_layers = 3
    output_size = 1
    device = torch.device('cuda')
    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00833417514)

    num_epochs = 75
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor.to(device))
        loss = criterion(outputs.squeeze(), y_train_tensor.to(device))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor.to(device))
        predicted_probs = torch.sigmoid(outputs).cpu().numpy()
        accuracy = backtest(X_test, predicted_probs, y_test)
        print(f'Accuracy: {accuracy}')

    torch.save(model.state_dict(), 'RNN_Model')


if __name__ == "__main__":
    data_preprocessing()
