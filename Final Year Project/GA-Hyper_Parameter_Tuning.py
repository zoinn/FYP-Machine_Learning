import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from Forex_Data import read_csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from geneticalgorithm import geneticalgorithm as ga
from RNN_LSTM import LSTM, backtest

data = read_csv().drop(0)
scaler = MinMaxScaler()
data[['open', 'high', 'low', 'close']] = scaler.fit_transform(data[['open', 'high', 'low', 'close']])
X = data[['open', 'high', 'low', 'close']].values
y = data['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_lstm(params):
    input_size, hidden_size, num_layers, output_size = map(int, params[:-1])
    learning_rate = float(params[-1])
    # print(input_size, hidden_size, num_layers, output_size, learning_rate)

    device = torch.device('cuda')

    model = LSTM(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.from_numpy(X_train).type(torch.Tensor).to(device)
    y_train_tensor = torch.from_numpy(y_train).type(torch.Tensor).to(device)

    num_epochs = 75
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs.squeeze(), y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.from_numpy(X_test).type(torch.Tensor).to(device)
        outputs = model(X_test_tensor)
        predicted_probs = torch.sigmoid(outputs).cpu().numpy()
        accuracy = backtest(X_test, predicted_probs, y_test)

    return -accuracy

if __name__ == "__main__":
    varbound = [
        [4, 4],  # input_size
        [1, 64],  # hidden_size
        [1, 8],  # num_layers
        [1, 1],  # output_size
        [0.001, 0.01]  # learning_rate
    ]

    modelGA = ga(function=evaluate_lstm, dimension=5, variable_type='real', variable_boundaries=np.array(varbound))

    modelGA.run()

    best_params = modelGA.output_dict['variable']
    best_accuracy = -modelGA.output_dict['function']
    print('Best parameters found by genetic algorithm:', best_params)
    print("Best accuracy:", best_accuracy)
    print()
