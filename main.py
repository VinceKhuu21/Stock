import torch
import pandas as pd

from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockTradesRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


from datetime import datetime
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.preprocessing import StandardScaler

#yoinked above code from documentation

#test data frame
df = pd.DataFrame(
    {
        "Name": [
            "bob", "joe",  "maclom"
        ],
        "Age": [22, 55,  33],
        "Sex": ["male", "female", "female"]
    }
)

print(df);
#api key get from alpaca info
API_KEY = "PKK7ERX7IQAUFQQ456AEYYHMU5"
SECRET_KEY = "GgfPsuTUEmwmq1r8fQVR5FAZkAeNBAkaYe5WouMoF8SV"
#, paper = True, remeber to do paper
stock_historical_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL", "MSFT"],
    timeframe=TimeFrame.Day,                 # Options: Day, Hour, Minute, etc.
    start=datetime(2026, 1, 1),              # Start date
    end=datetime(2026, 6, 1)                 # End date
)

bars = stock_historical_data_client.get_stock_bars(request_params)

df = bars.df
print(df.head(-3))

