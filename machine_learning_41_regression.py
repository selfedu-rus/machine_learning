# Случайный лес для задачи регрессии

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, np.pi, 0.1)
n_samples = len(x)
y = np.cos(x) + np.random.normal(0.0, 0.1, n_samples)
x = x.reshape(-1, 1)

clf = RandomForestRegressor(max_depth=2, n_estimators=4, random_state=1)
clf.fit(x, y)
yy = clf.predict(x)

plt.plot(x, y, label="cos(x)")
plt.plot(x, yy, label="DT Regression")
plt.grid()
plt.legend()
plt.title('Четыре дерева глубиной 2')
plt.show()
