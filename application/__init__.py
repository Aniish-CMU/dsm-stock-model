# import keras.layers
import os
import pandas as pd
import numpy as np
import requests
from flask import Flask, request, Response, json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


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
