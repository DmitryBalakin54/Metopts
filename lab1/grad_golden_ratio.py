import math

import numpy as np
import matplotlib.pyplot as plt

GOLDEN_CONST = (math.sqrt(5) - 2) / 2.0

eps = 0.000001
max_iter = 1000

left = 0.1
right = 7.0


def st(f, epsilon, *dot):
    res = []
    for ind, line in enumerate(np.eye(len(dot))):
        l = left
        r = right

        grd = grad_func(f)(*dot)
        for _ in range(max_iter):
            sign = grd[ind] / math.fabs(grd[ind]) if grd[ind] != 0 else 1

            x1 = l + (1 - GOLDEN_CONST) * (r - l)
            x2 = l + GOLDEN_CONST * (r - l)

            f_x1 = f(*(dot + x1 * line * sign))
            f_x2 = f(*(dot + x2 * line * sign))

            if f_x1 < f_x2:
                l = x2
            elif f_x2 < f_x1:
                r = x1
            else:
                break

            if math.fabs(r - l) < epsilon:
                break

        res.append((l + r) / 2)
    return np.array(res)


grad_func = lambda f: lambda *args: np.array(
    [(f(*(args + offset)) - f(*args)) / eps for offset in np.eye(len(args)) * eps])

next_dot_func = lambda f: lambda step, args: np.array(args) - step * grad_func(f)(*args)

calc_stop_func = lambda dot_1, dot_2: math.sqrt(sum((dot_1[i] - dot_2[i]) ** 2 for i in range(len(dot_1))))

stop_criteria_func = lambda dot_1, dot_2: calc_stop_func(dot_1, dot_2) < eps

step_func = lambda f: lambda *dot: st(f, eps, *dot)


def gradient_descent(f, *start_dot):
    next_dot = next_dot_func(f)
    step = step_func(f)
    dot = np.array(start_dot)
    for _ in range(max_iter):
        next_step = step(*dot)
        last_dot, dot = dot, next_dot(next_step, dot)
        if stop_criteria_func(last_dot, dot):
            break

    return dot


def run(f, *start_dot):
    return gradient_descent(f, *start_dot)


if __name__ == '__main__':
    print(run(lambda x, y: x ** 2 + y ** 2, 4, 4))
    print(run(lambda x, y: x ** 2 - 2 * x + y ** 2 + 1, 4, 4))
