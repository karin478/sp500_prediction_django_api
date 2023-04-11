import yfinance as yf

def fetch_stock_prices(company_names, days):
    stock_prices = []
    for name in company_names:
        stock_data = yf.download(name, period=f"{days}d", interval='1d')
        stock_prices.append(stock_data['Close'].values[-days:])
    return stock_prices

def fetch_sp500_prices(days):
    sp500_data = yf.download('^GSPC', period=f"{days}d", interval='1d')
    return sp500_data['Close'].values[-days:]
