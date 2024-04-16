import newton
import numpy as np
import math


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
    run_method(method, lambda x: x[0] ** 2 + x[1] ** 2, np.array([-5, 5]),
               'x^2 + y^2', log_iter, draw, history)
    run_method(method, lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, np.array([2, -4, 12]),
               'x^2 + y^2 + z^2', log_iter, draw, history)
    run_method(method, lambda x: x[0] ** 4 - 3 * x[0] ** 3 + 2 * x[0] ** 2 - x[0], np.array([1]),
               'x^4 - 3x^3 + 2x^2 -x', log_iter, draw, history)
    run_method(method, lambda x: math.sin(x[0]) ** 2 + 2 * math.cos(x[0]) ** 2, np.array([1]),
               'sin(x)^2 + 2 * cos(x)^2', log_iter, draw, history)
    run_method(method, lambda x: -math.log10(1.0 / math.fabs(x[0])) * x[0], np.array([2]),
               '-log_10(1 / |x|) * x', log_iter, draw, history)
    run_method(method, lambda x: (1 - x[0]) ** 2 + 2 * (x[1] - x[0] ** 2) ** 2, np.array([-6, 6]),
               '(1 - x)^2 + 2 * (y - x^2)^2', log_iter, draw, history)


solve(newton, log_iter=True, draw=True, history=True)
