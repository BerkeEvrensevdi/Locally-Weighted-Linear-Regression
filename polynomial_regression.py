import operator
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import math

m = 100
X = np.random.rand(m,1)*2
y = np.sin(2*math.pi*X)+np.random.randn(m,1)



polynomial_features= PolynomialFeatures(degree=9)
x_p = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_p, y)
y_pred = model.predict(x_p)


plt.plot(X, y, "b.")

sort_axis = operator.itemgetter(0)

sorted_zip = sorted(zip(X,y_pred), key=sort_axis)

X, y1_predict = zip(*sorted_zip)

plt.plot(X, y1_predict, color='m')

plt.show()