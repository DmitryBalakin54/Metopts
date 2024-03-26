import math

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


def run_f(gr, f, arg0, arg1, string, log):
    if log:
        gr.log_iter = True

    res = gr.run(f, arg0, arg1)
    res = [format(i, ".10f") for i in res]
    print(f'f = {string}, res = {res}', end='')
    if log:
        print(f', iters = {gr.it}', end='')
    print()


def run_grad(gr, arg0, arg1, log=True):
    print('-' * 3 + f'{gr.__name__} x = {arg0} y = {arg1}' + '-' * 3)

    run_f(gr, lambda x, y: x ** 2 + y ** 2, arg0, arg1, 'x^2 + y^2', log)
    run_f(gr, lambda x, y: x ** 2 - 2 * x + y ** 2 + 1, arg0, arg1,'x^2 - 2x + y^2 + 1', log)
    run_f(gr, lambda x, y: x ** 2 - 2 * x * y + y ** 2 + 1, arg0, arg1, 'x^2 - 2xy + y^2 + 1', log)
    run_f(gr, lambda x, y: math.sin(x) ** 2 + math.cos(y) ** 2, arg0, arg1, 'sin(x)^2 + cos(x)^2', log)


run_grad(grad, -100, 50)
run_grad(dech, -100, 50)
run_grad(gold, -100, 50)

print(end="\n\n\n")

run_grad(grad, 4, -3)
run_grad(dech, 4, -3)
run_grad(gold, 4, -3)
