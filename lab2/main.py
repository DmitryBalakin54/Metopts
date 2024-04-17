import newton
import numpy as np
import math

import newton_line_search


def run_method(method, f, x0, name, log_iter=False, draw=False, history=False):
    method.log_iter = log_iter
    method.name = name
    method.log_plot = draw
    method.log_history = history
    res = method.run(f, x0)

    print('-' * 4 + name + '-' * 4)
    print('Стартовая точка точка', x0)
    print('Минимум функции:', res['dot'])
    print('Значение функции в минимуме:', res['val_str'])

    if log_iter:
        print('iter= ', method.it)

    print('-' * (8 + len(method.name)))

    method.set_default()


def solve(method, log_iter=False, draw=False, history=False):
    print('-' * 8 + method.__name__ + '-' * 8)
    run_method(method, lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, np.array([2, -4, 12]),
               'x^2 + y^2 + z^2', log_iter, draw, history)

    run_method(method, lambda x: x[0] ** 2 + x[1] ** 2, np.array([-5, 5]),
               'x^2 + y^2', log_iter, draw, history)
    run_method(method, lambda x: (1 - x[0]) ** 2 + 2 * (x[1] - x[0] ** 2) ** 2, np.array([-6, 6]),
               '(1 - x)^2 + 2 * (y - x^2)^2', log_iter, draw, history)

    run_method(method, lambda x: x[0] ** 4 - 3 * x[0] ** 3 + 2 * x[0] ** 2 - x[0], np.array([1]),
               'x^4 - 3x^3 + 2x^2 -x', log_iter, draw, history)
    run_method(method, lambda x: math.sin(x[0]) ** 2 + 2 * math.cos(x[0]) ** 2, np.array([1]),
               'sin(x)^2 + 2 * cos(x)^2', log_iter, draw, history)
    run_method(method, lambda x: -math.log10(1.0 / math.fabs(x[0])) * x[0], np.array([2]),
               '-log_10(1 / |x|) * x', log_iter, draw, history)
    run_method(method, lambda x: x[0] ** 2 - 2 * x[0] - 2, np.array([7]),
               'x^2 - 2x - 2', log_iter, draw, history)
    run_method(method, lambda x: x[0] ** 3 + x[0] ** 2 - 2 * x[0] - 2, np.array([3]),
               'x^3 + x^2 - 2x - 2', log_iter, draw, history)
    print('-' * (16 + len(method.__name__)))


def qadr(method, log_iter=False, draw=False, history=False):
    print('-' * 8 + method.__name__ + '-' * 8)
    run_method(method, lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, np.array([2, -4, 12]),
               'x^2 + y^2 + z^2', log_iter, draw, history)

    run_method(method, lambda x: x[0] ** 2 + x[1] ** 2, np.array([-5, 5]),
               'x^2 + y^2', log_iter, draw, history)
    run_method(method, lambda x: (1 - x[0]) ** 2 + 2 * (x[1] - x[0] ** 2) ** 2, np.array([-6, 6]),
               '(1 - x)^2 + 2 * (y - x^2)^2', log_iter, draw, history)
    run_method(method, lambda x: x[0] ** 2 + 1000 * x[1] ** 2, np.array([3, 4]),
               'x^2 + 1000y^2', log_iter, draw, history)

    run_method(method, lambda x: 1000 * x[0] ** 2 - x[0] + 4, np.array([22]),
               '1000x^2 - x + 4', log_iter, draw, history)
    run_method(method, lambda x: x[0] ** 2 - 2 * x[0] - 2, np.array([7]),
               'x^2 - 2x - 2', log_iter, draw, history)

    print('-' * (16 + len(method.__name__)))


solve(newton, log_iter=True, draw=True, history=True)
#solve(newton_line_search, log_iter=True, history=True)
qadr(newton, log_iter=True, draw=True, history=True)
qadr(newton_line_search, log_iter=True, history=True)
