from io import StringIO
import requests
import pandas as pd
import numpy as np

def get_forex_data_CSV(startDate = '2023-01-01',ticker = 'eurusd',fileName = 'forex_training_data.csv'):
    headers = {
        'Content-Type': 'text/csv'
        'format=csv'
    }
    requestResponse = requests.get('https://api.tiingo.com/tiingo/fx/'+ticker+'/prices?startDate='+startDate+'&resampleFreq'
                                   '=1hour&token={API-TOKEN}', headers=headers)
    data_to_file_CSV(requestResponse.text,fileName)

def data_to_file_CSV(csvData, fileName):
    csvData = csvData.strip('[')
    csvData = csvData.strip(']')
    csvData = csvData.strip('"')
    csvData = csvData.strip('{')
    csvData = csvData.replace('{"date"','date')
    csvData = csvData.replace(':', ',')
    csvData = csvData.split('}')
    df = pd.read_csv(StringIO('\n'.join(csvData)))
    df.to_csv(fileName, index=False)


def format_CSV(fileName='forex_training_data.csv'):
    df = pd.read_csv(fileName, header=None)
    df = df.drop(0,axis=1)
    df = df.drop(2,axis=1)
    df = df.drop(4,axis=1)
    df = df.drop(6,axis=1)
    df = df.drop(8,axis=1)
    df = df.drop(10, axis=1)
    # Specify header names
    headers = ['date','ticker','open','high','low','close']

    df.columns = headers

    df.reset_index(drop=True, inplace=True)

    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%dT%H,%M,%S.%fZ')
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    df = df.drop(0) # Weird floating point error so need this to remove first data row
    df.to_csv(fileName, index=False)
    return df

def read_csv(fileName = 'forex_training_data.csv'):
    df = pd.read_csv(fileName)
    return df

def generate_test_data(date):
    get_forex_data_CSV(startDate=date,fileName='forex_test_data.csv')
    format_CSV(fileName='forex_test_data.csv')

