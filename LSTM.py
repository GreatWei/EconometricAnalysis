# import packages
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

# setting figure size
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10

# for normalizing data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# read the file
df = pd.read_csv('NSE-TATAGLOBAL.csv')

# print the head
df.head()

# setting index as date
df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
df.index = df['Date']

# creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
print("new_data===================================")
for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]

# setting index
new_data.index = new_data.Date
print(len(new_data))
new_data.drop('Date', axis=1, inplace=True)
print(new_data.shape)
print("new_data===================================")
# creating train and test sets
dataset = new_data.values
# print(dataset)
train = dataset[0:987, :]
print("train: ", len(train))
valid = dataset[987:, :]
print("valid: ", len(valid))
# converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    # print("x_train: ",scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])
    # print("y_train: ",scaled_data[i, 0])
print("x_train, y_train===================================")
print(np.shape(x_train))
print(np.shape(y_train))

x_train, y_train = np.array(x_train), np.array(y_train)
print("x_train")

print(np.shape(x_train))
print(np.shape(y_train))
print(x_train.shape[1])
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# predicting 246 values, using past 60 from the train data
#                    2035               987
print("input start:", len(new_data) - len(valid) - 60)
inputs = new_data[len(new_data) - len(valid) - 60:].values
print("input values", inputs)
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)
print("X_test===================================")
X_test = []
print("input :", inputs.shape)
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
print(X_test.shape)
print("X_test===================================")
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print("X_test:",X_test)
print(X_test.shape)
# print(X_test[0].shape)


# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
print("closing_price===========================================")
print(X_test.shape)

closing_price = model.predict(X_test)
closing_price = np.array(closing_price)
print(closing_price)
print(closing_price.shape)
print("===========================================")
closing_price = scaler.inverse_transform(closing_price)

# for plotting
train = new_data[:987]
valid = new_data[987:987 + len(closing_price)]
valid['Predictions'] = closing_price
# valid['MyPredictions'] = myclosing_price
# plt.plot(new_data,label='new_data')
ax = plt.gca()
from matplotlib.dates import AutoDateLocator, DateFormatter, DayLocator

# # ??????x??????????????????
# alldays = mdates.DayLocator()  # ??????????????????
# ax1.xaxis.set_major_locator(alldays)  # ???????????????
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m%d'))
# # ?????????????????????
# hoursLoc = mpl.dates.HourLocator(interval=6)  # ???6?????????1?????????
# ax1.xaxis.set_minor_locator(hoursLoc)
# ax1.xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
# # ??????pad??????????????????????????????????????????
# ax1.tick_params(pad=10)
# 0.36053395
autodates = AutoDateLocator()
alldays = DayLocator()  # ??????????????????
ax.xaxis.set_major_locator(alldays)
yearsFmt = DateFormatter('%Y-%m-%d')
ax.xaxis.set_major_formatter(yearsFmt)
# print("train.index:", train.index)
# print("valid.index:", valid.index)
print("valid",valid)
plt.plot(train['Close'], label='train_Close')
plt.plot(valid['Close'], label='Close')
plt.plot(valid['Predictions'], label='Predictions')
plt.legend(loc='best')
plt.show()
