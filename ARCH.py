from scipy import stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arch
import tushare as ts

IndexData = ts.get_k_data(code='sh', start='2014-01-01', end='2016-08-01');
IndexData.index = pd.to_datetime(IndexData.date);
close = IndexData.close;
rate = (close - close.shift(1)) / close.shift(1)
data = pd.DataFrame();
data['rate'] = rate;
data = data.dropna();
data1 = np.array(data['rate'])
# data['rate'].plot(figsize=(15, 5))
# t = sm.tsa.stattools.adfuller(data1)
# print("p-vaule:", t[1])
# fig = plt.figure(figsize=(20, 5))
# ax1 = fig.add_subplot(111)
# fig = sm.graphics.tsa.plot_acf(data1, lags=20, ax=ax1)

# order = (8, 0)
# model = sm.tsa.ARMA(data1, order).fit()
# print(model)
# at = data1 - model.fittedvalues
# at2 = np.square(at)
# plt.figure(figsize=(10, 6))
# plt.plot(at, label='at')
# plt.legend();
# plt.subplot(212)
# plt.plot(at2, label='at^2')
# plt.legend(loc=0);

train = data[:-10]
test = data[-10:]
# am = arch.arch_model(train, mean='AR', lags=8, vol='ARCH', p=4)
# res = am.fit()
# print(res.summary())
#
# print(res.hedgehog_plot())
# pre = res.forecast(horizon=10, start=619).iloc[619]
# plt.figure(figsize=(10, 4))
# plt.plot(test, label='realValue')
# pre.plot(label='predictValue')
# plt.plot(np.zeros(10), label='zero')
# plt.legend(loc=0)
GARCH = arch.arch_model(train, mean='AR', lags=8, vol='GARCH', p=4)
res=GARCH.fit()
print(GARCH.fit().summary())
res.plot()
plt.plot(data1)
res.hedgehog_plot()


print(res.params)
ini = res.resid[-8:]
a = np.array(res.params[1:9])
w = a[::1]
# for i in range(10):
#     new = test[i]-(res.params[0]+w.dot(ini[-8:]))
#     ini = np.append(ini,new)
# print(len(ini))
#
# at_pre=ini[-10:]
# at_pre2=at_pre**2
# #print(at_pre2)
# ini2 = res.conditional_volatility[-2:]
# for i in range(10):
#     new = 0.


plt.show()