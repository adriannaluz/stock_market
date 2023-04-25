import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# loading .env file
load_dotenv()

# Vanguard Total Stock Market Index Fund (VTI)
# loading VTI data
path = os.getenv("data_path")
data = pd.read_csv(path + "VTI.csv")
data_cols = list(data.columns)

# fig, ax = plt.subplots(figsize=(18,9))
# sns.set_style("ticks")
# sns.lineplot(data=data,x="Date", y='High',color='firebrick')
# sns.despine()
# plt.xticks(range(0,data.shape[0],200),data['Date'].loc[::200],rotation=45)
# plt.title('VTI Stock Market',fontsize=16)
# plt.xlabel('Date',fontsize=12)
# plt.ylabel('High',fontsize=12)
# plt.show()

# Scaling of the training data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data['High'].values.reshape(-1, 1))

# Splitting the data into training and test dataset
train_data, test_data = train_test_split(data_scaled)

def split_sequence(sequence, window):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + window
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

window_size = 60
features = 1

X_train, y_train = split_sequence(data_scaled, window_size)
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],features)

# Define the LSTM model
model_lstm = Sequential()
model_lstm.add(LSTM(units=125, activation="tanh", input_shape=(window_size, features)))
model_lstm.add(Dense(25))
model_lstm.add(Dense(units=1))

model_lstm.compile(optimizer='adam', loss='mse')

model_lstm.summary()

model_lstm.fit(X_train, y_train, epochs=15, batch_size=32)

dataset_total = data.loc[:,"High"]
inputs = dataset_total[len(dataset_total) - len(test_data) - window_size :].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test, y_test = split_sequence(inputs, window_size)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], features)
predicted_stock_price = model_lstm.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)
y_test = scaler.inverse_transform(y_test)

def plot_predictions(test, predicted):
    plt.plot(test, color="gray", label="Real")
    plt.plot(predicted, color="red", label="Predicted")
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()


def return_rmse(test, predicted):
    rmse = np.sqrt(mean_squared_error(test, predicted))
    print("The root mean squared error is {:.2f}.".format(rmse))

plot_predictions(y_test,predicted_stock_price)

return_rmse(y_test,predicted_stock_price)