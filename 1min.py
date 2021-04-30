import pandas as pd
import numpy as np

# to plot within notebook
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pandas import Timestamp

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# setting figure size
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10

# for normalizing data
from sklearn.preprocessing import MinMaxScaler


def trainModel(x_train, y_train, units):
    model = Sequential()
    # model.add(LSTM(units=150, return_sequences=True, input_shape=(x_train.shape[1], 7)))
    # model.add(LSTM(units=150))
    # model.add(LSTM(units=270, return_sequences=True, input_shape=(x_train.shape[1], 7)))
    # model.add(LSTM(units=270))
    model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 7)))
    model.add(LSTM(units=units))
    model.add(Dense(7))

    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.fit(x_train, y_train, epochs=4, batch_size=20, verbose=2)
    # model.fit(x_train, y_train, epochs=4, batch_size=30, verbose=2)
    # epochs=7, batch_size=15
    # epochs=9, batch_size=13
    # epochs=9, batch_size=11
    # epochs=10, batch_size=11
    model.fit(x_train, y_train, epochs=11, batch_size=11, verbose=2)
    # model.save();
    print(model.summary())
    return model


def predictval(model, original_data, leng, scaler):
    all_closing_price = []
    original_data_cp = original_data
    for i in range(0, leng):
        # print("i:",i)
        X_test = []
        inputs = scaler.transform(original_data_cp)
        # print(inputs.shape)
        for j in range(0, i + 1):
            X_test.append(inputs[j:(j + 60), ])
        X_test = np.array(X_test)
        # print(X_test.shape)
        closing_price = model.predict(X_test)
        tmp = scaler.inverse_transform(closing_price)
        all_closing_price = tmp
        original_data_cp = np.append(original_data, tmp)
        original_data_cp = original_data_cp.reshape(-1, 7)

    return all_closing_price


tf.random.set_seed(54294)
scaler = MinMaxScaler(feature_range=(0, 1))
# read the file
df = pd.read_csv('1minethusdt.csv')
df.head()
print(df)
df.index = (df['Id'] - 1619494800) / 60
data = df.sort_index(ascending=True, axis=0)
data.drop('Id', axis=1, inplace=True)
print(data)
dataset = data.values
print("dataset", dataset.shape)
train = dataset[0:1500, :]
valid = dataset[1500:, :]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data.shape)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, ])
    y_train.append(scaled_data[i,])

x_train, y_train = np.array(x_train), np.array(y_train)
print("x_train", x_train.shape)
# print("x_train",x_train)
print("y_train", y_train.shape)

inputs = dataset[1440:, :]
original_data = inputs[0:60]
leng = inputs.shape[0] - 60
inputs = scaler.transform(inputs)
X_test = []
print("inputs :", inputs.shape)
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, ])
X_test = np.array(X_test)

# create and fit the LSTM network
model = trainModel(x_train, y_train,132)

my_closing_price = predictval(model, original_data, leng, scaler)
my_closing_price = np.array(my_closing_price)
print("closing_price===========================================")
print(X_test.shape)

closing_price = model.predict(X_test)
closing_price = np.array(closing_price)
closing_price = scaler.inverse_transform(closing_price)

# print("closing_price",closing_price)
# print("closing_price",closing_price[:,3])

train = data[:1500]['Close']
valid = data[1500:1500 + len(closing_price)]
valid['Predictions'] = closing_price[:, 3]
valid['MyPredictions'] = my_closing_price[:, 3]
# plt.plot(train, label='train_Close')
plt.plot(valid['Close'], label='Close')
plt.plot(valid['Predictions'], label='Predictions')
plt.plot(valid['MyPredictions'], label='MyPredictions')
plt.legend(loc='best')
plt.show()


def My(x_train, y_train, X_test):
    # create and fit the LSTM network
    for units in range(130, 200):
        model = trainModel(x_train, y_train, units)

        my_closing_price = predictval(model, original_data, leng, scaler)
        my_closing_price = np.array(my_closing_price)
        print("closing_price===========================================")
        print(X_test.shape)

        closing_price = model.predict(X_test)
        closing_price = np.array(closing_price)
        closing_price = scaler.inverse_transform(closing_price)

        # print("closing_price",closing_price)
        # print("closing_price",closing_price[:,3])

        train = data[:1500]['Close']
        valid = data[1500:1500 + len(closing_price)]
        valid['Predictions'] = closing_price[:, 3]
        valid['MyPredictions'] = my_closing_price[:, 3]
        # plt.plot(train, label='train_Close')
        plt.plot(valid['Close'], label='Close')
        plt.plot(valid['Predictions'], label='Predictions')
        plt.plot(valid['MyPredictions'], label=('MyPredictions' + units))
        plt.legend(loc='best')
        plt.show()


My(x_train, y_train, X_test)
