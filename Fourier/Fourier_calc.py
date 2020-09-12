import numpy as np
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
import seaborn
from Invoice import Invoice
from Enterprise import Enterprise
from datetime import date
import pandas as pd

frame = pd.read_csv("./frame.csv")

Y = frame['资金'].loc[frame['企业代号'] == 'E7']
Y : np.ndarray = Y.to_numpy()

Y = Y / np.sum(Y)


X = range(Y.size)


transformed = np.abs(fft(Y))
print(transformed.size)
transformed.sort()
print(transformed[-2])
print(np.mean(transformed))

plt.plot(X, transformed)
plt.show()
