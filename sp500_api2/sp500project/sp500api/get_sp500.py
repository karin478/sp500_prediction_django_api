# 导入yfinance模块
import yfinance as yf

# 定义股票代码
stock_code = "^GSPC"

# 创建一个Ticker对象
stock = yf.Ticker(stock_code)

# 获取昨天的日期
from datetime import date, timedelta
yesterday = date.today() - timedelta(days=1)

# 下载历史数据，从2015年1月1日到昨天，以每日为间隔
data = stock.history(start="2012-01-01", end=yesterday, interval="1d")

# 打印数据
print(data)
import requests

#response = requests.get("https://finance.yahoo.com/quote/OXY?p=OXY")
#print(response.status_code)
#print(response.text)