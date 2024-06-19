import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from Forex_Data import read_csv
from sklearn.preprocessing import MinMaxScaler
from Forex_News import print_news
from RNN_LSTM import LSTM

# Code reuse due to scaler problems
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
        trade_return = 10 if (pred == actual) else -10
        balance += trade_return
        total_trades += 1
        if pred == actual:
            correct_trades += 1

    accuracy = correct_trades / total_trades

    print(f':: Backtest results :: \nTotal Trades : {total_trades}, Correct Trades : {correct_trades}, Balance : {balance}')

    return accuracy

def load_model():
    model = LSTM(4, 42, 3, 1)
    model.load_state_dict(torch.load('RNN_Model_84'))
    model.to(device)
    return model

class ForexAdvisor:
    def print_news(self):
        # Function to print news
        print("Printing latest news...")
        print_news()

    def evaluate_data(self, givenFile = ""):
        #Function to evaluate data and print graph
        global untestedData
        print("Evaluating given data with the model...")
        if not givenFile:
            with torch.no_grad():
                outputs = model(X_new_tensor.to(device))
                predictedProbs = torch.sigmoid(outputs).cpu().numpy()
                accuracy = backtest(untestedData, predictedProbs, y_new)
                print(f'Accuracy on new data: {accuracy}')

            with torch.no_grad():
                predictedTensor = model(X_new_tensor.to(device))
                predictedCloseScaled = predictedTensor.cpu().numpy().reshape(-1, 1)
                dataCopy = untestedData.copy()
                dataCopy['close'] = predictedCloseScaled
                dataCopy[['open', 'high', 'low', 'close']] = scaler.inverse_transform(
                    dataCopy[['open', 'high', 'low', 'close']])
                dataCopy['target'] = np.where(dataCopy['close'].shift(-1) > dataCopy['close'], 1, 0)

            combined = pd.concat([dataCopy['date'].head(60), dataCopy['target'].head(60), untestedData['target'].head(60)],
                                 axis=1)
            combined.columns = ['date', 'T1', 'T2']

            # Plots step graph showing accuracy
            plt.figure(figsize=(12, 8))
            plt.step(x=combined['date'], y=combined['T1'], where='mid', color='blue', marker=None, linewidth=1,
                     label='Predicted Direction over time')
            plt.step(x=combined['date'], y=combined['T2'], where='mid', color='red', marker=None, linewidth=1,
                     linestyle='--',
                     label='Actual Direction over time')
            plt.title('Predicted Direction vs Actual Direction')
            plt.xlabel('Date')
            plt.ylabel('Direction of Price (1 = Up, 0 = Down)')
            plt.xticks(combined['date'], rotation=45, ha='right')
            plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
            plt.yticks([0, 1])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            untestedData = read_csv(fileName=givenFile)
            ohl = untestedData[['open', 'high', 'low']].values
            close = untestedData['close'].values.reshape(-1, 1)

            ohlScaler = MinMaxScaler()
            ohlScaled = ohlScaler.fit_transform(ohl)

            closeScaler = MinMaxScaler()
            closeScaled = closeScaler.fit_transform(close)

            X_scaled = np.concatenate((ohlScaled, closeScaled), axis=1)

            X_tensor = torch.from_numpy(X_scaled).type(torch.Tensor)

            with torch.no_grad():
                outputs = model(X_tensor.to(device))
                predictedProbs = torch.sigmoid(outputs).cpu().numpy()
            predicted = closeScaler.inverse_transform(predictedProbs.reshape(-1, 1))
            targets = []
            for x in range(len(predicted)):
                try:
                    targets.append(np.where((predicted[x] - predicted[x+1]>0),1,0))
                except:
                    print()
            xAxis = range(len(targets))
            plt.figure(figsize=(12, 8))
            plt.step(xAxis, targets, where='mid', color='blue', marker=None, linewidth=1,
                     label='Predicted direction over time')
            plt.title('Predicted Directions over time')
            plt.xlabel('Time (+Hours)')
            plt.ylabel('Direction of Price (1 = Up, 0 = Down)')
            plt.xticks(xAxis, labels=xAxis, ha='right')
            plt.yticks([0, 1])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def exit_program(self):
        # Function to exit the program
        print("Exiting...")
        quit()

    # Main menu function
    def main_menu(self):
        print("\nForex Advisor Tool")
        print("1. Print News")
        print("2. Evaluate Data")
        print("3. Exit")

        choice = input("Enter your choice (1-3): ")

        return choice

    # Main function
    def main(self):
        while True:
            choice = self.main_menu()
            match choice:
                case '1':
                    self.print_news()
                case '2':
                    file = input("Enter filename/path:")
                    self.evaluate_data(givenFile=file)
                case '3':
                    self.exit_program()
                case _:
                    print("Invalid choice. Please enter a number between 1 and 3.")

if __name__ == "__main__":
    device = torch.device("cuda")
    untestedData = read_csv(fileName='forex_test_data.csv')
    scaler = MinMaxScaler()
    untestedData[['open', 'high', 'low', 'close']] = scaler.fit_transform(untestedData[['open', 'high', 'low', 'close']])
    X_new = untestedData[['open', 'high', 'low', 'close']].values
    y_new = untestedData['target'].values
    X_new_tensor = torch.from_numpy(X_new).type(torch.Tensor)

    model = load_model()
    advisor = ForexAdvisor()
    advisor.main()

