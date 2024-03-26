import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

step = 0.1
eps = 0.000001
max_iter = 10000

log_iter = False
it = 0

log_plot = False
counter = 0
name = ''

grad_func = (
    lambda f: lambda *args: np.array([(f(*(args + offset)) - f(*args)) / eps for offset in np.eye(len(args)) * eps]))

next_dot_func = (lambda f: lambda args: np.array(args) - step * grad_func(f)(*args))

calc_stop_func = (lambda dot_1, dot_2: math.sqrt(sum((dot_1[i] - dot_2[i]) ** 2 for i in range(len(dot_1)))))

stop_criteria_func = (lambda dot_1, dot_2: calc_stop_func(dot_1, dot_2) < eps)


def gradient_descent(f, *start_dot):
    global it

    if log_iter:
        it = 1

    next_dot = next_dot_func(f)

    last_dot = np.array(start_dot)
    dot = next_dot(last_dot)

    for _ in range(max_iter):
        if log_iter:
            it += 1

        last_dot, dot = dot, next_dot(dot)
        if stop_criteria_func(last_dot, dot):
            break

    return dot


def run(f, *start_dot):
    res = gradient_descent(f, *start_dot)

    if log_plot:
        global counter, name
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')

        xl = res[0] - math.fabs(res[0] - start_dot[0])
        xr = res[0] + math.fabs(res[0] - start_dot[0])
        yl = res[1] - math.fabs(res[1] - start_dot[1])
        yr = res[1] + math.fabs(res[1] - start_dot[1])

        X = np.arange(xl, xr,  (xr - xl) / 100.0)
        Y = np.arange(yl,   yr, (yr - yl) / 100.0)
        X, Y = np.meshgrid(X, Y)
        Z = f(X, Y)

        ax.plot_surface(X, Y, Z, alpha=0.4, color='b')
        plt.savefig(f'figs/grad_learning_rate_{name}_{counter}.png', dpi=400)
        plt.close('all')
        name = ''
        counter += 1
    return res


if __name__ == '__main__':
    print(run(lambda x, y: x ** 2 + y ** 2, 10, -10))
    print(run(lambda x, y: x ** 2 - 2 * x + y ** 2 + 1, -30, 40))
