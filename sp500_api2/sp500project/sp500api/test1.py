# Import yfinance module
import yfinance as yf

# Define the symbol and the date range
symbol = "AAPL"
start = "2012-01-01"
end = "2022-12-13"

# Fetch the historical data
data = yf.download(symbol, start, end)

# Print the data
print(data)

import requests

response = requests.get('https://api.github.com')

print(response.text)

import requests

symbol = 'AAPL'
start = '2012-01-01'
end = '2022-12-13'
interval = '1d'
url = f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}?period1={start}&period2={end}&interval={interval}&events=history&includeAdjustedClose=true'

response = requests.get(url)

if response.status_code == 200:
    print(response.text)
else:
    print(f"Error {response.status_code}: {response.text}")

from yahoo_fin.stock_info import get_data

symbol = 'AAPL'
start = '2012-01-01'
end = '2022-12-13'
interval = '1d'
data = get_data(symbol, start_date=start, end_date=end, index_as_date=False, interval=interval)
print(data)

import requests

url = 'https://query1.finance.yahoo.com/v7/finance/download/AAPL?period1=1325376000&period2=1640995200&interval=1d&events=history&crumb=tO1hNZoUQeQ'

response = requests.get(url)

if response.status_code == 200:
    print('Connection successful')
else:
    print(f'Connection failed: Error {response.status_code}: {response.text}')