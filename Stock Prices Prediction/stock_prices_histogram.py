import matplotlib.pyplot as plt
import pandas_datareader.data as web

# Define company
stock = ['AAPL']

# Define date of stock prices
start_date = '01/01/2001'
end_date = '01/01/2017'

# Read data from web
data = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)['Adj Close'] # Adjusted closing price

# Calculate daily returns
daily_returns = (data/ data.shift(1)) - 1

# Show collected data on histogram
daily_returns.hist(bins=100)
plt.show()