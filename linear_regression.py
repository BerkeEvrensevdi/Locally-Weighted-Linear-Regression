import numpy as np
import operator
import matplotlib.pyplot as plt


def linear_regression(X, y, iterNo, eta):

    theta = np.random.randn(2,1)

    for iteration in range(iterNo):
        gradient = (2 / m) * X.T.dot(X.dot(theta) - y)

        theta = theta - eta*gradient

    return theta

m=100;
X=np.random.rand(m,1)*2
y=100+3*X+np.random.randn(m,1)
X_b = np.c_[np.ones((len(X),1)),X]

theta = linear_regression(X_b,y,1000,0.1)

y_predict = theta[0] + theta[1]*X


plt.plot(X, y, "b.")

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(X,y_predict), key=sort_axis)

X, y1_predict = zip(*sorted_zip)

plt.plot(X, y1_predict, color='m')

plt.show()