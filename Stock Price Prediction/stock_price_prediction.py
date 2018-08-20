#Reccurrent Neural Network



# Data Preprocessing

#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the training set
train_df = pd.read_csv('Google_Stock_Price_Train.csv')
train_set = train_df.iloc[:,1:2].values

#Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
train_set_scaled = sc.fit_transform(train_set)

#Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(train_set_scaled[i-60:i,0])
    y_train.append(train_set_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)

#Reshaping the dataset
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))



# Building the RNN model

#Importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Initializing the RNN
regressor = Sequential()

#Adding the first LSTM layer and Dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

#Adding second, third and fourth LSTM and Dropout layers
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units=1))

#Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

#Fitting the RNN to the training set
regressor.fit(X_train, y_train, batch_size=32, epochs=100)



# Making predictions and visualizing results

#Getting real stock price of 2017
#Importing the test set
test_df = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_prices = test_df.iloc[:,1:2].values

#Getting predictions
concat_df = pd.concat((train_df['Open'], test_df['Open']), axis=0)
inputs = concat_df[len(concat_df) - len(test_df) - 60:].values
inputs = inputs.reshape(-1,1)

inputs = sc.transform(inputs)

X_test = []

for i in range(60, 80):
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

predictions = regressor.predict(X_test)

predictions = sc.inverse_transform(predictions)

#Visualizing results
plt.plot(real_stock_prices, color='red', label='Real stock prices')
plt.plot(predictions, color='blue', label='Predictions')
plt.title('Google stock price predictions')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
