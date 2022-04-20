# Алгоритм регрессии AdaBoost на решающих деревьях

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

np.random.seed(123)

x = np.arange(0, np.pi/2, 0.1).reshape(-1, 1)
y = np.sin(x) + np.random.normal(0, 0.1, x.shape)

# plt.plot(x, y)
# plt.grid()
# plt.show()

T = 5                   # число алгоритмов в композиции
max_depth = 2           # максимальная глубина решающих деревьев
algs = []               # список из полученных алгоритмов
s = np.array(y.ravel())
for n in range(T):
    # создаем и обучаем решающее дерево
    algs.append(DecisionTreeRegressor(max_depth=max_depth))
    algs[-1].fit(x, s)

    s -= algs[-1].predict(x)    # пересчитываем остатки


# восстанавливаем исходный сигнал по набору полученных деревьев
yy = algs[0].predict(x)
for n in range(1, T):
    yy += algs[n].predict(x)

# отображаем результаты в виде графиков
plt.plot(x, y)      # исходный график
plt.plot(x, yy)     # восстановленный график
plt.plot(x, s)      # остаточный сигнал
plt.grid()
plt.show()
