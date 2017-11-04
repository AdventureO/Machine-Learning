from datetime import datetime

import numpy as np
import pandas_datareader.data as web
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


def create_dataset(stock, start_date, end_date, lags=5):

    # Fetch the stock data from Yahoo Finance
    data = web.DataReader(stock, data_source='yahoo', start=start_date, end=end_date)

    # Create a new data frame
    # We want to use additional features: lagged returns, today's returns and etc
    data['Today'] = data['Adj Close']

    # Create the lagged percentage returns columns
    for i in range(0, lags):
        data['Lag%s' % str(i + 1)] = data['Lag%s' % str(i + 1)].pct_change() * 100

    # "Direction" column (+1 or -1) indicating an up/down day
    data['Direction'] = np.sign(data['Today'])

    # Because of the shifts there are None values
    # We want to get rid of those
    data.drop(data.index[:6], inplace=True)

    return data


if __name__ == '__main__':

    # Create a lagged series of the S&P500  US stock market index
    data = create_dataset("AAPL", datetime(2015 , 1, 1), datetime(2017, 5, 31), lags=5)

    # Use the prior of two days of returns as predictor
    # Values, with direction as the response
    X = data[["Lag1", "Lag2", "Lag3", "Lag4"]]
    Y = data["Direction"]

    # The test data is split into two parts: before and after 1st Jan 2005
    start_test = datetime(2017, 1, 1)

    # Create training ans test data
    X_train =  X[X.index < start_test]
    X_test = X[X.index >= start_test]
    Y_train = Y[Y.index < start_test]
    Y_test = Y[Y.index >= start_test]

    # We use Logistic regression as machine learning model
    model = LogisticRegression()

    # Train the model with train data
    model.fit(X_train, Y_train)

    # Make predictions on the test data
    prediction = model.predict(X_test)

    # Output the hit-rate and confusion matrix for the model
    print("Accuracy of logistic regression model: %0.3f" % model.score(X_test, Y_test))
    print("Confusion matrix: \n%s" % confusion_matrix(prediction, Y_test))
