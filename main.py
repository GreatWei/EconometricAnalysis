# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from scipy import io
import pandas as pd
import pandas_datareader.data as web
import datetime
import statsmodels.api as sm
import quandl
import opendatatools.stock as st
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
# df,msg=st.get_quote('600000.SH,000002.SZ')
# print((df))
start = datetime.datetime(2017,1,1)
end = datetime.datetime(2019,9,26)
df1 = web.DataReader("F",'yahoo',start,end)
df1.head()
print(df1)
print("yahoo")
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
