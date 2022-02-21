# Расчет коэффициентов полиномиальной модели

import numpy as np


def predict_poly(x, koeff):
    res = 0
    xx = [x ** (len(koeff) - n - 1) for n in range(len(koeff))]

    for i, k in enumerate(koeff):
        res += k * xx[i]

    return res


x = np.arange(0, 10.1, 0.1)
y = 1 / (1 + 10 * np.square(x))

x_train, y_train = x[::2], y[::2]

N = len(x)

z_train = np.polyfit(x_train, y_train, 10)
print(z_train)
