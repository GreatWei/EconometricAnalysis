import pandas as pd
import numpy as np

tmp =[[12],[456]]
tmp = np.array(tmp)
tmp = tmp.reshape(-1, 1)
#print(tmp)
tmp=np.append(tmp,np.array([[11]]))
tmp = tmp.reshape(-1, 1)
print(tmp[1:,])


for i in range(1,4):
    print(i)