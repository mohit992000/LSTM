import yfinance as yf
import pandas as pd

stock_data = yf.download('AAPL' , start='2010-01-01' , end='2024-09-01' )
print(stock_data.head())
stock_data.to_csv("APPL_STOCK_DATA.CSV")

stock_data.fillna(method='ffill' , inplace=True)

stock_data['50_MA'] = stock_data["Close"].rolling(window=50).mean()
stock_data['200_MA'] = stock_data['Close'].rolling(window=2).mean()

print(stock_data.tail())

from sklearn.preprocessing import MinMaxScaler 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(stock_data[['Close']])

scaled_data_df = pd. DataFrame(scaled_data, columns=['Close_scaled'], index=stock_data.index)
stock_data = pd.concat([stock_data, scaled_data_df], axis=1)
print (stock_data.head())

import numpy as np

def create_sequences(data, time_steps=60):
    x=[]
    y=[]
    for i in range(time_steps, len(data)):
        x.append(data[i-time_steps:i, 0])
        y.append(data[i, 0])
        return np.array(x), np.array(y)
    X,y= create_sequences(scaled_data, time_steps=60)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    from keras.models import seqentalial 
    from keras.layers import Dense, LSTM, Droupout
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(Droupout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Droupout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X, y, epochs=5, batch_size=32)

    prediction = model.predict(X)
    predicted_prices = scaler.inverse_transform(prediction)

    import matplotlib.pyplot as plt

    plt.plot(stock_data.index[-len(predicted_prices):], predicted_prices, label='predicted price')
    plt.plot(stock_data.index[-len(predicted_prices):], stock_data['close'][-len(predicted_prices):], label='actual prices')
    plt.legend()
    plt.show()

    import talib # type: ignore
    # Add RSI
    stock_data['RSI'] = talib.RSI(stock_data['Close'], timeperiod=14)

    # Add MACD
    stock_data['MACD'], stock_data['MACD_Signal'], stock_data['MACD_Hist'] = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Add Bollinger Bands 
    stock_data['Upper_Band'], stock_data['Middle_Band'], stock_data['Lower_Band'] = talib.BRANDS(stock_data['Close'], timeperiod=20, nbdeup=2, nbdevdn=2)

    #check the new data with technical indicators
    print(stock_data.tail()) 


import statsmodels.api as sm
import matplotlib.pyplot as plt

# Autocorrelation and Partial Autocorrelation
fig, ax = plt.subplots(2, 1, figsize=(12,8))
sm.graphics.tsa.plot_acf(stock_data['Close'], lags=50, ax=ax[0])
sm.graphics.tsa.plot_pacf(stock_data['Close'], lags=50, ax=ax[1])
plt.show()

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional

# Advanced LSTM model
model = Sequential()

#First Bidirectional LSTM layers
model.add(Bidirectional(LSTM(units=100, return_sequences=True, input_shape=(X. shape[1], 1))))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))

#output layer
model.add(Dense(unit=1))

#compile the model with a lower learning rate
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with more epochs
model.fit(X, y, epochs=20, batch_size=32)


from keras.callbacks import EarlyStopping

#Early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor= 'val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])


# split data into train and test sets (80% train, 20% test)
train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:len(scaled_data)]

# prepare the test data
X_test, y_test = create_sequences(test_data, time_steps=60)

# Reshape X_test for LSTM input
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Make predictions
predicted_stock_prices = model.predict(X_test)

# Inverse the scaling to get actual stock prices
predicted_stock_prices = scaler.inverse_transform(predicted_stock_prices)
y_test_actual = scaler.inverse_transform([y_test])

# plot the predictions vs actual
plt.plot(y_test_actual[0], color='blue', label='Actual Stock price')
plt.plot(predicted_stock_prices, color='red', label='predicted Stock price')
plt.title('Stock price prediction')
plt.xlabel('Time')
plt.ylabel('Stock price')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_actual[0], predicted_stock_prices))
mae = mean_absolute_error(y_test_actual[0], predicted_stock_prices)

print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')









            



    
    
                       






                        