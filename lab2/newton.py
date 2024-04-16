import math

import numpy as np
import scipy.optimize
from matplotlib.ticker import LinearLocator
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt

eps = 1e-6
max_iter = 100

log_iter = False
it = 0

log_plot = False

log_history = False
history = []
levels = 10

name = ''
counter = 0


def set_default():
    global eps, max_iter, log_iter, it, log_history, log_plot, levels, history
    eps = 1e-6
    max_iter = 100
    log_iter = False
    it = 0
    log_plot = False
    log_history = False
    levels = 10
    history = []


def newton_method(f, x0):
    global it, log_iter, log_history, history
    cur_x = x0
    for _ in range(max_iter):
        if log_iter:
            it += 1

        if log_plot or log_history:
            history.append(cur_x)

        grad = approx_fprime(cur_x, f, epsilon=1e-6)
        hessian = approx_fprime(cur_x, lambda x: approx_fprime(x, f, epsilon=1e-6), epsilon=1e-6)

        if len(grad) > 1:
            new_x = cur_x - np.dot(np.linalg.linalg.inv(hessian), grad)
        else:
            new_x = cur_x - grad / hessian

        if np.linalg.norm(new_x - cur_x) < eps:
            if log_plot or log_history:
                history.append(new_x)

            return new_x

        cur_x = new_x

    if log_plot or log_history:
        history.append(cur_x)

    return cur_x


def get_abs_dot(d1, d2, offset):
    dot = np.array([0.0 for _ in d1])
    for i in range(len(d1)):
        dot[i] = d1[i] \
            if math.fabs(d1[i] - offset[i]) > math.fabs(d2[i] - offset[i]) \
            else d2[i]

    return dot


def get_max_dot(center, start):
    res = start
    if not log_history:
        return np.array(res)

    res = get_abs_dot(res, res, center)
    for dot in history:
        res = get_abs_dot(res, dot, center)

    return res


def get_X_Y_Z(start, finish, f):
    xl = finish[0] - math.fabs(finish[0] - start[0])
    xr = finish[0] + math.fabs(finish[0] - start[0])
    yl = finish[1] - math.fabs(finish[1] - start[1])
    yr = finish[1] + math.fabs(finish[1] - start[1])

    offset = math.fabs(xr - xl) / 10

    xl = (xr + xl) / 2 - math.fabs(xr - xl) / 2 - offset
    xr = (xr + xl) / 2 + math.fabs(xr - xl) / 2 + offset
    yl = (yr + yl) / 2 - math.fabs(yr - yl) / 2 - offset
    yr = (yr + yl) / 2 + math.fabs(yr - yl) / 2 + offset

    X = np.arange(xl, xr, (xr - xl) / 100.0)
    Y = np.arange(yl, yr, (yr - yl) / 100.0)
    X, Y = np.meshgrid(X, Y)
    Z = f(np.array([X, Y]))

    return X, Y, Z


def get_X_Y(start, finish, f):
    xl = min(start, finish)
    xr = max(start, finish)

    offset = math.fabs(xr - xl)

    xl = xl - offset
    xr = xr + offset

    X = np.linspace(xl, xr, 1000)
    Y = np.zeros_like(X)
    for i, x in enumerate(X):
        Y[i] = f(x)

    return X, Y


def make_new_name():
    return name.replace('*', '').replace('/', 'div').replace('|', 'm')


def draw_1D(start, finish, f):
    global name, counter
    X, Y = get_X_Y(start, finish, f)

    plt.plot(X, Y, color='b')
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')
    new_name = make_new_name()
    plt.savefig(f'figs/newton_{new_name}_{counter}.png', dpi=400)
    plt.close('all')
    counter += 1


def draw_2D(start, finish, f):
    global name, counter
    X, Y, Z = get_X_Y_Z(start, finish, f)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')

    ax.plot_surface(X, Y, Z, alpha=0.4, color='b')
    new_name = make_new_name()
    plt.savefig(f'figs/newton_{new_name}_{counter}.png', dpi=400)
    plt.close('all')
    counter += 1


def draw_way_1D(start, finish, f):
    global name, counter
    X, Y = get_X_Y(start, finish, f)

    plt.plot(X, Y, color='b')
    plt.scatter(history, [f(i) for i in history])
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('y')
    new_name = make_new_name()
    plt.savefig(f'figs/newton_levels_{new_name}_{counter}.png', dpi=400)
    plt.close('all')
    counter += 1


def draw_way_2D(start, finish, f):
    global counter, name
    X, Y, Z = get_X_Y_Z(start, finish, f)
    plt.contour(X, Y, Z, levels=levels)
    plt.scatter([i[0] for i in history], [i[1] for i in history])

    plt.grid(True)
    new_name = make_new_name()
    plt.savefig(f'figs/newton_levels_{new_name}_{counter}.png')
    counter += 1
    plt.close('all')


def run(f, x0):
    global it, log_iter, name, counter, history
    np.set_printoptions(suppress=True)

    dim = len(x0)

    x_min = newton_method(f, x0)
    f_x_min = f(x_min)
    f_x_min_str = np.format_float_positional(f_x_min)

    if log_plot and dim == 1:
        draw_1D(get_max_dot(x_min, x0), x_min, f)
    if log_plot and dim == 2:
        draw_2D(get_max_dot(x_min, x0), x_min, f)
    if log_history and dim == 1:
        draw_way_1D(get_max_dot(x_min, x0), x_min, f)
    if log_history and dim == 2:
        draw_way_2D(get_max_dot(x_min, x0), x_min, f)

    history = []
    name = ''

    return {'dot': x_min, 'val': f_x_min, 'val_str': f_x_min_str}


if __name__ == '__main__':
    log_iter = True

    name = 'x^2 + y^2'
    run(lambda x: x[0] ** 2 + x[1] ** 2, np.array([1, 1]))

    name = 'x^2 + y^2 + z^2'
    run(lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, np.array([2, -4, 12]))

    name = 'x^4 - 3x^3 + 2x^2 -x'
    run(lambda x: x[0] ** 4 - 3 * x[0] ** 3 + 2 * x[0] ** 2 - x[0], np.array([1]))

    name = 'sin(x)^2 + 2 * cos(x)^2'
    run(lambda x: math.sin(x[0]) ** 2 + 2 * math.cos(x[0]) ** 2, np.array([1]))

    name = '-log_10(1 / |x|) * x'
    run(lambda x: -math.log10(1.0 / math.fabs(x[0])) * x[0], np.array([2]))

    name = '(1 - x)^2 + 2 * (y - x^2)^2'
    run(lambda x: (1 - x[0]) ** 2 + 2 * (x[1] - x[0] ** 2) ** 2, np.array([3, -4]))

    set_default()
