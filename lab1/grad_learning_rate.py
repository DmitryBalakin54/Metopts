import math

import numpy as np
import matplotlib.pyplot as plt

step = 0.01
eps = 0.000001
max_iter = 10000

_grad = lambda f: lambda *args: np.array([(f(*(args + offset)) - f(*args)) / eps for offset in np.eye(len(args)) * eps])

_next_dot = lambda f: lambda args: np.array(args) - step * _grad(f)(*args)

_calc_stop = lambda dot_1, dot_2: math.sqrt(sum((dot_1[i] - dot_2[i]) ** 2 for i in range(len(dot_1))))

_stop_criteria = lambda dot_1, dot_2: _calc_stop(dot_1, dot_2) < eps


def _gradient_descent(f, *start_dot):
    next_dot = _next_dot(f)

    last_dot = np.array(start_dot)
    dot = next_dot(last_dot)

    for _ in range(max_iter):
        last_dot, dot = dot, next_dot(dot)
        if _stop_criteria(last_dot, dot):
            break

    return dot


def run(f, *start_dot):
    return _gradient_descent(f, *start_dot)


if __name__ == '__main__':
    print(run(lambda x, y: x ** 2 + y ** 2, 1, 1))
    print(run(lambda x, y: x ** 2 - 2 * x + y ** 2 + 1, 3, 4))
