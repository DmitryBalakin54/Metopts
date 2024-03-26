import grad_learning_rate as grad
import grad_dichotomy as dech
import grad_golden_ratio as gold


grad.step = 0.09
grad.eps = 0.00000001
grad.max_iter = 10000

dech.eps = 0.000001
dech.max_iter = 1000
dech.left = 0.001 / 10
dech.right = 7.0 / 10

gold.eps = 0.000001
gold.max_iter = 1000
gold.left = 1.0 / 10
gold.right = 7.0 / 1


def run_grad(gr, x, y):
    print('-' * 3 + f'{gr.__name__}' + '-' * 3)
    res_1 = gr.run(lambda x, y: x ** 2 + y ** 2, x, y)
    res_1 = [format(i, ".10f") for i in res_1]
    print(res_1)

    res_2 = gr.run(lambda x, y: x ** 2 - 2 * x + y ** 2 + 1, x, y)
    res_2 = [format(i, ".10f") for i in res_2]
    print(res_2)

    res_3 = gr.run(lambda x, y: x ** 2 - 2 * x * y + y ** 2 + 1, x, y)
    res_3 = [format(i, ".10f") for i in res_3]
    print(res_3)


run_grad(grad, -100, 50)
run_grad(dech, -100, 50)
run_grad(gold, -100, 50)
