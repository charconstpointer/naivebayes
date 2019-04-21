from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt
import numpy as np


my_data = genfromtxt('data.csv', delimiter=',')

X = my_data[:, 0:2]
y = my_data[:, 2:]
y = y.ravel()
nb = GaussianNB()
nb.fit(X, y)
print(nb.predict([[183, 95]]))
