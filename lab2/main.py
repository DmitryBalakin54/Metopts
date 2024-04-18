import numpy as np
import math

from newton import Newton
from newton_line_search import NewtonLineSearch
from newton_wolfie import NewtonWolfie


def run_method(class_method, f, x0, name):
    res = class_method.run(f, x0, name)

    print('-' * 4 + name + '-' * 4)
    print('Стартовая точка точка', x0)
    print('Минимум функции:', res['dot'])
    print('Значение функции в минимуме:', res['val_str'])

    if class_method.log_iter:
        print('iter= ', res['it'])

    print('-' * (8 + len(name)))


def solve(class_method):
    print('-' * 64 + class_method.__class__.__name__ + '-' * 64)
    run_method(class_method, lambda x: x[0] ** 2 + x[1] ** 2 + x[2] ** 2, np.array([2, -4, 12]),
               'x^2 + y^2 + z^2')

    run_method(class_method, lambda x: x[0] ** 2 + x[1] ** 2, np.array([-5, 5]),
               'x^2 + y^2')
    run_method(class_method, lambda x: (1 - x[0]) ** 2 + 2 * (x[1] - x[0] ** 2) ** 2, np.array([3, 30]),
               '(1 - x)^2 + 2 * (y - x^2)^2')

    run_method(class_method, lambda x: x[0] ** 4 - 3 * x[0] ** 3 + 2 * x[0] ** 2 - x[0], np.array([1]),
               'x^4 - 3x^3 + 2x^2 -x')
    run_method(class_method, lambda x: math.sin(x[0]) ** 2 + 2 * math.cos(x[0]) ** 2, np.array([1]),
               'sin(x)^2 + 2 * cos(x)^2')
    run_method(class_method, lambda x: -math.log10(1.0 / math.fabs(x[0])) * x[0], np.array([2]),
               '-log_10(1 / |x|) * x')
    run_method(class_method, lambda x: x[0] ** 2 - 2 * x[0] - 2, np.array([7]),
               'x^2 - 2x - 2')
    run_method(class_method, lambda x: x[0] ** 3 + x[0] ** 2 - 2 * x[0] - 2, np.array([3]),
               'x^3 + x^2 - 2x - 2')
    print('-' * (128 + len(class_method.__class__.__name__)), end='\n\n')


eps = 1e-6
max_iter = 1000
solve(Newton(log_iter=True, log_plot=False, log_history=False, eps=eps, max_iter=max_iter))
solve(NewtonLineSearch(log_iter=True, log_plot=False, log_history=False, left=0.01, right=10.0, eps=eps, max_iter=max_iter))
solve(NewtonWolfie(log_iter=True, log_plot=False, log_history=False, eps=eps, max_iter=max_iter))
