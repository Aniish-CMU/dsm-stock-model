# import keras.layers
import os
import pandas as pd
import numpy as np
import requests
from flask import Flask, request, Response, json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
# from keras.models import Sequential
# from keras.layers import LSTM
# from keras.layers import Dense

# def get_apikey(key_name, json_filename='secret.json'):
#     BASE_DIR = Path(__file__).resolve().parent
#     json_filepath = os.path.join(BASE_DIR, json_filename)
#
#     if(not os.path.isfile(json_filepath)):
#         print("JSON File Not Found")
#         raise FileNotFoundError
#
#     with open(json_filepath) as f:
#         json_p = json.loads(f)
#
#     try:
#         value=json_p[key_name]
#         return value
#     except KeyError:
#
#         error_msg = "ERROR: Unvalid Key"
#         return error_msg

def get_historical_data(symbol, start_date = None):
    api_key = open(r'api_key.txt')
    api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=full'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df[f'Time Series (Daily)']).T
    df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. adjusted close': 'adj close', '6. volume': 'volume'})
    for i in df.columns:
        df[i] = df[i].astype(float)
    df.index = pd.to_datetime(df.index)
    df = df.iloc[::-1].drop(['7. dividend amount', '8. split coefficient'], axis = 1)
    if start_date:
        df = df[df.index >= start_date]
    df.to_csv(symbol+".csv")
    return df

app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    ticker = data['ticker']
    #valid_tickers = ['MVRS', 'AMZN', 'AAPL']

    #if ticker not in valid_tickers:
    #    return Response(json.dumps({'error': 'Invalid ticker symbol. Please enter MVRS for Meta, AMZN for Amazon, or AAPL for Apple.'}), status=400)
    
    df = get_historical_data(data['ticker'], "2019-01-01")
    average = np.average(df['open'])
    #median = np.median(df['open'])
    #stdev = np.std(df['open'])
    response = {
        'ticker': ticker,
        'average_open_price': average
    }
    return Response(json.dumps(average))
if __name__ == "__main__":
    app.run()
#
# def get_historical_data(symbol, start_date = None):
#     api_key = open(r'api_key.txt')
#     api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=full'
#     raw_df = requests.get(api_url).json()
#     df = pd.DataFrame(raw_df[f'Time Series (Daily)']).T
#     df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. adjusted close': 'adj close', '6. volume': 'volume'})
#     for i in df.columns:
#         df[i] = df[i].astype(float)
#     df.index = pd.to_datetime(df.index)
#     df = df.iloc[::-1].drop(['7. dividend amount', '8. split coefficient'], axis = 1)
#     if start_date:
#         df = df[df.index >= start_date]
#     return df
#
# def result(ticker):
#     return get_historical_data(ticker, '2020-01-01')
#
# msft_hist = result('MSFT')
# #print(msft_hist)
#
#
# dataset_train = msft_hist
# #print(dataset_train.head())
#
# #Use the open stock price column to train the model
# training_set = dataset_train.iloc[:,1:2].values
# price = dataset_train.iloc[:,1].values
#
# average_price = (sum(price)/len(price))
# sum_price = sum(price)
# #print(sum_price)
# print(average_price)
#print(training_set)
#print(training_set.shape)

# Normalzing the dataset

#scaler = MinMaxScaler(feature_range = (0,1))
#scaled_training_set = scaler.fit_transform(training_set)

##print(scaled_training_set)

#Creating X-train and t_train data structure

# X_train = []
# y_train = []
# last_train_date = training_set.shape[0]-30
# for i in range(60, last_train_date):
#     X_train.append(scaled_training_set[i-60:i, 0])
#     y_train.append(scaled_training_set[i, 0])
# X_train = np.array(X_train)
# y_train = np.array(y_train)
#
# #print(X_train)
#print(y_train)

#print(X_train.shape)
#print(y_train.shape)

#Reshape the data
#X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
#print(X_train.shape)

# Building the Model by Importing the Crucial Libraries and Adding Different Layers to LSTM

# regressor = Sequential()
#
# regressor.add(LSTM(units = 50, return_sequences= True, input_shape= (X_train.shape[1], 1)))
# regressor.add(tf.keras.layers.Dropout(rate=0.2))
#
# regressor.add(LSTM(units = 50, return_sequences= True))
# regressor.add(tf.keras.layers.Dropout(rate=0.2))
#
# regressor.add(LSTM(units = 50, return_sequences= True))
# regressor.add(tf.keras.layers.Dropout(rate=0.2))
#
# regressor.add(LSTM(units = 50))
# regressor.add(tf.keras.layers.Dropout(rate=0.2))
#
# regressor.add(Dense(units=1))

#Fitting the Model

# regressor.compile(optimizer= 'adam', loss= 'mean_squared_error')
# regressor.fit(X_train, y_train, epochs =100, batch_size = 32)

#Extracting the Actual Stock Prices
# dataset_test = training_set[-30:]
# print(dataset_test)
