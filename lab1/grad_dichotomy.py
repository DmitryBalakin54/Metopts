import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

eps = 0.000001
max_iter = 1000

left = 0.0001
right = 0.07

log_iter = False
it = 0

log_plot = False
log_history = False
plot_scale = 1
history = []
counter = 0
name = ''


def st(f, epsilon, *dot):
    res = []
    grd = grad_func(f)(*dot)
    for ind, line in enumerate(np.eye(len(dot))):
        sign = grd[ind] / math.fabs(grd[ind]) if grd[ind] != 0 else 1
        l = left
        r = right

        for _ in range(max_iter):
            m = (l + r) / 2

            f_l = f(*(dot + l * line * sign))
            f_r = f(*(dot + r * line * sign))
            f_m = f(*(dot + m * line * sign))

            if f_m < f_l:
                r = m
            elif f_m < f_r:
                l = m
            else:
                break

            if math.fabs(r - l) < epsilon:
                break
        res.append((l + r) / 2)
    return np.array(res)


grad_func = (lambda f: lambda *args: np.array(
    [(f(*(args + offset)) - f(*args)) / eps for offset in np.eye(len(args)) * eps]))
next_dot_func = (lambda f: lambda step, args: np.array(args) - step * grad_func(f)(*args))

calc_stop_func = (lambda dot_1, dot_2: math.sqrt(sum((dot_1[i] - dot_2[i]) ** 2 for i in range(len(dot_1)))))

stop_criteria_func = (lambda dot_1, dot_2: calc_stop_func(dot_1, dot_2) < eps)

step_func = (lambda f: lambda *dot: st(f, eps, *dot))


def gradient_descent(f, *start_dot):
    global it, history

    if log_iter:
        it = 1

    next_dot = next_dot_func(f)
    step = step_func(f)
    last_dot = np.array(start_dot)

    if log_history:
        history.append(last_dot)

    dot = np.array(start_dot)

    if log_history:
        history.append(dot)

    for _ in range(max_iter):
        if log_iter:
            it += 1
        next_step = step(*dot)
        last_dot, dot = dot, next_dot(next_step, dot)

        if log_history:
            history.append(dot)

        if stop_criteria_func(last_dot, dot):
            break

    return dot


def run(f, *start_dot):
    global counter, name, history
    history = []
    res = gradient_descent(f, *start_dot)

    if log_history or log_plot:
        xl = res[0] - math.fabs(res[0] - start_dot[0])
        xr = res[0] + math.fabs(res[0] - start_dot[0])
        yl = res[1] - math.fabs(res[1] - start_dot[1])
        yr = res[1] + math.fabs(res[1] - start_dot[1])

        xl = (xr + xl) / 2 - plot_scale * math.fabs(xr - xl) / 2
        xr = (xr + xl) / 2 + plot_scale * math.fabs(xr - xl) / 2
        yl = (yr + yl) / 2 - plot_scale * math.fabs(yr - yl) / 2
        yr = (yr + yl) / 2 + plot_scale * math.fabs(yr - yl) / 2

        X = np.arange(xl, xr, (xr - xl) / 100.0)
        Y = np.arange(yl, yr, (yr - yl) / 100.0)
        X, Y = np.meshgrid(X, Y)
        Z = f(X, Y)

    if log_plot:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter('{x:.02f}')

        ax.plot_surface(X, Y, Z, alpha=0.4, color='b')
        plt.savefig(f'figs/grad_dichotomy_{name}_{counter}.png', dpi=400)
        plt.close('all')
        counter += 1

    if log_history:
        plt.contour(X, Y, Z, levels=15)
        plt.scatter([i[0] for i in history], [i[1] for i in history])
        plt.grid(True)
        plt.savefig(f'figs/grad_dichotomy_levels_{name}_{counter}.png')
        counter += 1
        plt.close('all')

    name = ''
    return res


if __name__ == '__main__':
    print(run(lambda x, y: x ** 2 + y ** 2, -1, -1))
    print(run(lambda x, y: x ** 2 - 2 * x + y ** 2 + 1, -2, -1))
