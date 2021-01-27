import numpy as np
import operator
import matplotlib.pyplot as plt
import math

def weighted_linear_regression(X, y, iteration_cnt, eta, x, tau):

    theta = np.random.randn(2, 1)
    row, col = np.shape(X)
    weights = np.zeros((row,1))


    for i in range(row):
        diff = x - X[i]
        weights[i] = np.exp(diff.dot(diff.T) / (-2 * tau ** 2))

    #print(weights)
    for iteration in range(iteration_cnt):
        gradient = (2 / m) * (weights*X).T.dot(X.dot(theta)-y)
        theta = theta - eta * gradient


    return theta


m = 100
X = np.random.rand(m, 1) * 2
y = np.sin(2 * math.pi * X) + np.random.randn(m, 1)

X_b = np.c_[np.ones((len(X), 1)), X]

y_pred = np.zeros(len(X_b))


for i in range(len(X_b)):
    theta = weighted_linear_regression(X_b, y, 100, 0.4, X_b[i], 0.001)
    y_pred[i] = theta[0] + theta[1] * X[i]


plt.plot(X, y, "b.")

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(X,y_pred), key=sort_axis)

X, y1_predict = zip(*sorted_zip)

plt.plot(X, y1_predict, color='m')

plt.show()