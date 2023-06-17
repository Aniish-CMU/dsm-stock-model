# import keras.layers
import os
import pandas as pd
import numpy as np
import requests
from flask import Flask, request, Response, json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from prophet import Prophet


def data_preprocess(start_time, raw_data):
    '''Takes starting time of the series in yyyy-mm-dd format, the raw
        datafile and data frequency - daily, weekly or monthly
    '''
    datetime_object = datetime.strptime(start_time, '%Y-%m-%d')
    frequency_encoding_dict = {'daily':'D', 'weekly':'W', 'monthly':'M'}
    date_lst = (pd.date_range(start=datetime_object, periods=len(raw_data), \
        freq="D"))
    df_ts = pd.DataFrame({'ds':date_lst.values, 'y':raw_data['adj close']})
    return df_ts

def get_historical_data(symbol, start_date = None):
    df = pd.read_csv(symbol+".csv")
    #api_key = open(r'api_key.txt')
   # api_url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=full'
    #companyname_dict = {"Apple": "AAPL", "Amazon": "AMZN", "Meta": "META"}
    
    #raw_df = requests.get(api_url).json()
   # df = pd.DataFrame(raw_df[f'Time Series (Daily)']).T
    #df = df.rename(columns = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close', '5. adjusted close': 'adj close', '6. volume': 'volume'})
   # for i in df.columns:
   #     df[i] = df[i].astype(float)
   # df.index = pd.to_datetime(df.index)
   # df = df.iloc[::-1].drop(['7. dividend amount', '8. split coefficient'], axis = 1)

   # if start_date:
       #  df = df[df.index >= start_date]
   # df.to_csv(symbol+".csv")
    df_prophet = data_preprocess("2019-01-02", df)
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=20)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']][-20:]

app = Flask(__name__)

@app.route('/api', methods=['GET', 'POST'])
def predict():
    #get data from request
    data = request.get_json(force=True)
    ticker = data['ticker']
    #valid_tickers = ['MVRS', 'AMZN', 'AAPL']

    #if ticker not in valid_tickers:
    #    return Response(json.dumps({'error': 'Invalid ticker symbol. Please enter MVRS for Meta, AMZN for Amazon, or AAPL for Apple.'}), status=400)
    
    df = get_historical_data(ticker , "2019-01-01")
    #median = np.median(df['open'])
    #stdev = np.std(df['open'])
    response = {
        'ticker': ticker,
        '20_day_average': np.average(df['yhat']),
        '20_day_median': np.median(df['yhat']),
        '20_day_high': np.max(df['yhat']),
        '20_day_low': np.min(df['yhat']),
    }
    return Response(json.dumps(response))
if __name__ == "__main__":
    app.run()