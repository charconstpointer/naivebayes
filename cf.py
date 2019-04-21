from sklearn.naive_bayes import GaussianNB
from numpy import genfromtxt


my_data = genfromtxt('data.csv', delimiter=',')

X = my_data
y = [1, 1, 0, 0, 0, 0, 1, 0]

nb = GaussianNB()
nb.fit(X, y)
print(nb.predict([[153, 45]]))
