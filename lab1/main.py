import math
import numpy as np

import grad_learning_rate as grad
import grad_dichotomy as dech
import grad_golden_ratio as gold

grad.step = 0.09
grad.eps = 0.00000001
grad.max_iter = 10000

dech.eps = 0.00000001
dech.max_iter = 1000
dech.left = 0.01 / 100
dech.right = 7.0 / 100

gold.eps = 0.00000001
gold.max_iter = 1000
gold.left = 1.0 / 10
gold.right = 7.0 / 10


def run_f(gr, f, arg0, arg1, name, log, log_plt):
    if log:
        gr.log_iter = True
    if log_plt:
        gr.log_plot = True
        gr.name = name
    res = gr.run(f, arg0, arg1)
    res = [format(i, ".10f") for i in res]
    print(f'f = {name}, res = {res}', end='')
    if log:
        print(f', iters = {gr.it}', end='')
    print()


def run_grad(gr, arg0, arg1, log=False, log_plt=False):
    print('-' * 3 + f'{gr.__name__} x = {arg0} y = {arg1}' + '-' * 3)

    run_f(gr, lambda x, y: x ** 2 + y ** 2, arg0, arg1, 'x^2 + y^2', log, log_plt)
    run_f(gr, lambda x, y: x ** 2 - 2 * x + y ** 2 + 1, arg0, arg1,'x^2 - 2x + y^2 + 1', log, log_plt)
    run_f(gr, lambda x, y: x ** 2 - 2 * x * y + y ** 2 + 1, arg0, arg1, 'x^2 - 2xy + y^2 + 1', log, log_plt)
    run_f(gr, lambda x, y: np.sin(x) ** 2 + np.cos(y) ** 2, arg0, arg1, 'sin(x)^2 + cos(x)^2', log, log_plt)


run_grad(grad, -100, 50, True, True)
run_grad(dech, -100, 50, True)
run_grad(gold, -100, 50, True)

print(end="\n\n\n")

run_grad(grad, 4, -3, True, True)
run_grad(dech, 4, -3, True)
run_grad(gold, 4, -3, True)
