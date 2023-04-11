import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import joblib
from .lstm_model import LSTMModel
from yahoo_fin import stock_info as si
import os


import requests

def download_file_from_github(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)

# GitHub 存储库中文件的 URL
fitted_scalers_url = "https://raw.githubusercontent.com/karin478/sp500_LSTM_prediction_model/main/fitted_scalers.pkl"
model_weights_url = "https://raw.githubusercontent.com/karin478/sp500_LSTM_prediction_model/main/model_weights.pth"

# 下载目标路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
# 下载目标路径
fitted_scalers_destination = os.path.join(current_dir, 'fitted_scalers.pkl')
model_weights_destination = os.path.join(current_dir, 'model_weights.pth')


# 下载文件
download_file_from_github(fitted_scalers_url, fitted_scalers_destination)
download_file_from_github(model_weights_url, model_weights_destination)





import torch
import pickle

input_size = 51
hidden_size = 512
num_layers = 4
output_size = 1
dropout_rate = 0.2
bidirectional = True

def load_model1(model_path, scalers_path):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    return model, scalers


# Replace these lines in your code
# torch.save(model.state_dict(), 'model_weights.pth')
# with open('fitted_scalers.pkl', 'wb') as f:
#    pickle.dump(scalers, f)

# Add these lines to load the model and scalers

# Now, the rest of your code can use the loaded model and scalers
import torch
import pickle


def load_model(model_path, scalers_path):
    model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)

    return model, scalers

import os
import torch

# 获取当前 Python 文件所在的绝对路径
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

# 模型文件名
model_weights_filename = "model_weights.pth"
model_scaler_filename = "fitted_scalers.pkl"
# 将当前路径与模型文件名组合
model_weights_filepath = os.path.join(current_dir, model_weights_filename)
model_scaler_filename_filepath = os.path.join(current_dir, model_scaler_filename)
model_weights_path = model_weights_filepath
fitted_scalers_path = model_scaler_filename_filepath



#model_weights_path = '/Users/lvbinghan/Desktop/CS/sp500_api2/sp500project/sp500api/model_weights.pth'
#fitted_scalers_path = '/Users/lvbinghan/Desktop/CS/sp500_api2/sp500project/sp500api/fitted_scalers.pkl'
model, scalers = load_model(model_weights_path, fitted_scalers_path)

# 获取股票数据
tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "META", "V", "UNH", "XOM", "JNJ", "WMT", "JPM", "PG", "MA", "LLY", "HD", "HD","CVX","ABBV","MRK","AVGO","KO","PEP","ORCL","PFE","BAC","COST","TMO", "CSCO","MCD","NKE","CRM","DHR","DIS","ABT", "ACN", "ADBE", "LIN", "UPS", "TXN", "AMD", "VZ", "CMCSA", "NEE", "PM", "WFC", "MS", "BMY", "NFLX"]
start_date = "2012-01-04"
end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

companies_close = []
min_length = float('inf')

# 直接下载股票数据并获取收盘价
# 直接下载股票数据并获取收盘价
for ticker in tickers:
    stock_data = si.get_data(ticker, start_date=start_date, end_date=end_date)
    if stock_data.empty:
        print(f"Skipping {ticker}: No data found for this date range.")
        continue
    close = stock_data['close'].values
    close = close.reshape(-1, 1)
    min_length = min(min_length, len(close))
    companies_close.append(close)



# 获取S&P 500数据
sp500_data = si.get_data('^GSPC', start_date=start_date, end_date=end_date)
sp500_close = sp500_data['close'].values
sp500_close = sp500_close.reshape(-1, 1)



# 截断S&P 500数据以匹配最小长度
sp500_close = sp500_close[:min_length]

# 截断公司收盘价数据以匹配最小长度
truncated_companies_close = [close[:min_length] for close in companies_close]
companies_close = np.hstack(truncated_companies_close)

# 整合S&P 500和股票数据
all_data = np.hstack((sp500_close, companies_close))

# 对数据进行缩放
scalers = [MinMaxScaler(feature_range=(0, 1)) for _ in range(all_data.shape[1])]
scaled_data = np.hstack([scalers[i].fit_transform(all_data[:, i].reshape(-1, 1)) for i in range(all_data.shape[1])])


def create_time_series(data, window, data2, scaler_y):
    x, y = [], []
    for i in range(window, len(data)):
        x.append(data[i - window:i])
        y.append(scaler_y.transform(data2[i].reshape(-1, 1))[0])
    return np.array(x), np.array(y)

window = 30
x_all, y_all = create_time_series(scaled_data, window, sp500_close, scalers[0])

train_size = int(len(x_all) * 0.8)
x_train, x_test = x_all[:train_size], x_all[train_size:]
y_train, y_test = y_all[:train_size], y_all[train_size:]

# x_train, y_train = create_time_series(train_data, window)
# x_test, y_test = create_time_series(test_data, window)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()  # Only use the first column (S&P 500) as the target

batch_size = 32

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 获取当前 Python 文件所在的绝对路径
#current_file_path = os.path.abspath(__file__)
#current_dir = os.path.dirname(current_file_path)

# 模型文件名
model_weights_filename = "model_weights.pth"
model_scaler_filename = "fitted_scalers.pkl"
# 将当前路径与模型文件名组合
model_weights_filepath = os.path.join(current_dir, model_weights_filename)
model_scaler_filename_filepath = os.path.join(current_dir, model_scaler_filename)
model_weights_path = model_weights_filepath
fitted_scalers_path = model_scaler_filename_filepath

#model_weights_path = '/Users/lvbinghan/Desktop/CS/sp500_api2/sp500project/sp500api/model_weights.pth'
#fitted_scalers_path = '/Users/lvbinghan/Desktop/CS/sp500_api2/sp500project/sp500api/fitted_scalers.pkl'
model, scalers = load_model(model_weights_path, fitted_scalers_path)



#model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_rate, bidirectional)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# Your training and evaluation code here


model.eval()
predictions = []

with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs[0].float()
        outputs = model(inputs)
        predictions.extend(outputs.numpy().flatten())

# Create a zero matrix with the same shape as test_data
#predicted_data = np.zeros_like(test_dataset)
# Create a zero matrix with the same shape as the test data
predicted_data = np.zeros((len(predictions), 1))


# Assign the model's predictions to the first column of the matrix
predicted_data[:len(predictions), 0] = np.array(predictions).reshape(-1, 1)[:, 0]

# Perform inverse scaling on the predicted_data matrix
predicted_prices = np.hstack(
    [scalers[i].inverse_transform(predicted_data[:, i].reshape(-1, 1)) for i in range(predicted_data.shape[1])])

# Only take the first column (S&P 500) as the predicted prices
predicted_prices = predicted_prices[:, 0]

#plt.figure(figsize=(24, 12))
#plt.plot(pd.to_datetime(sp500_data.index[-len(y_test):]), sp500_close[-len(y_test):], color='blue', label='Actual S&P 500 Price')
#plt.plot(pd.to_datetime(sp500_data.index[-len(y_test):]), predicted_prices[:len(y_test)], color='red', label='Predicted S&P 500 Price')


#plt.xlabel('Date')
#plt.ylabel('S&P 500 Price')
#plt.legend()
#plt.show()


def predict_next_day(model, previous_days, scalers, window):
    previous_days_scaled = np.array(
        [scalers[i].transform(previous_days[:, i].reshape(-1, 1)) for i in range(previous_days.shape[1])]).T
    input_data = torch.tensor(previous_days_scaled[-window:]).view(1, -1, previous_days.shape[1]).float()
    with torch.no_grad():
        predicted_price = model(input_data).item()
    return scalers[0].inverse_transform([[predicted_price]])[0][0]


def predict_next_n_days(model, initial_data, scalers, window, n):
    predictions = []
    for _ in range(n):
        next_day_price = predict_next_day(model, initial_data, scalers, window)
        initial_data = np.append(initial_data[1:], [np.concatenate(([next_day_price], initial_data[-1, 1:]))], axis=0)
        predictions.append(next_day_price)
    return predictions


initial_data = all_data[-window:]

n_days = 30
future_predictions = predict_next_n_days(model, initial_data, scalers, window, n_days)


last_date = pd.to_datetime(sp500_data.index[-1])
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)
future_dates = future_dates.strftime('%Y-%m-%d').tolist()

last_year_dates = sp500_data.index[-365:].values

#plt.figure(figsize=(24, 12))
#plt.plot(pd.to_datetime(last_year_dates), sp500_close[-365:], color='blue', label='Actual S&P 500 Price (last 1 year)')
#plt.plot(pd.to_datetime(future_dates), future_predictions, color='red', label='Predicted S&P 500 Price (next 30 days)')
#plt.xlabel('Date')
#plt.ylabel('S&P 500 Price')
#plt.legend()
#plt.xticks(rotation=45)
#plt.show()

print(future_predictions)

def predictor():

    return future_predictions

def update_stock_data():
    global end_date, sp500_close, companies_close, all_data, scaled_data, x_all, y_all, x_train, x_test, y_train, y_test

    # 更新结束日期为昨天
    end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')

    # 使用新的结束日期重新获取股票数据
    companies_close = []
    min_length = float('inf')
    for ticker in tickers:
        stock_data = si.get_data(ticker, start_date=start_date, end_date=end_date)
        if stock_data.empty:
            print(f"Skipping {ticker}: No data found for this date range.")
            continue
        close = stock_data['close'].values
        close = close.reshape(-1, 1)
        min_length = min(min_length, len(close))
        companies_close.append(close)

    # 使用新的结束日期重新获取S&P 500数据
    sp500_data = si.get_data('^GSPC', start_date=start_date, end_date=end_date)
    sp500_close = sp500_data['close'].values
    sp500_close = sp500_close.reshape(-1, 1)

    # 更新数据变量
    sp500_close = sp500_close[:min_length]
    truncated_companies_close = [close[:min_length] for close in companies_close]
    companies_close = np.hstack(truncated_companies_close)
    all_data = np.hstack((sp500_close, companies_close))
    scaled_data = np.hstack([scalers[i].fit_transform(all_data[:, i].reshape(-1, 1)) for i in range(all_data.shape[1])])
    x_all, y_all = create_time_series(scaled_data, window, sp500_close, scalers[0])
    train_size = int(len(x_all) * 0.8)
    x_train, x_test = x_all[:train_size], x_all[train_size:]
    y_train, y_test = y_all[:train_size], y_all[train_size:]

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).float()



