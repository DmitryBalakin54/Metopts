import math
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

it = 0
eps = 0.000001
max_iter = 10000
log_iter = False
log_plot = False
log_history = False
plot_scale = 1
history = []
counter = 0
name = ''


def ob(f, X):
    if log_history:
        history.append(X)
    return f(*X)


def run(f, *start_dot):
    global counter, history, it

    dot = np.array(start_dot)
    res = opt.minimize(lambda X: ob(f, X), dot, method='Nelder-Mead', options={'maxiter' : max_iter}, tol=0.000001)
    it = res['nit']
    # opt.s
    # print(res)


    if log_history or log_plot:
        xl = res['x'][0] - math.fabs(res['x'][0] - start_dot[0])
        xr = res['x'][0] + math.fabs(res['x'][0] - start_dot[0])
        yl = res['x'][1] - math.fabs(res['x'][1] - start_dot[1])
        yr = res['x'][1] + math.fabs(res['x'][1] - start_dot[1])

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
        plt.savefig(f'figs/grad_golden_ratio_{name}_{counter}.png', dpi=400)
        plt.close('all')
        counter += 1

    if log_history:
        plt.contour(X, Y, Z, levels=15)
        plt.scatter([i[0] for i in history], [i[1] for i in history])
        plt.grid(True)
        plt.savefig(f'figs/grad_golden_ratio_levels_{name}_{counter}.png')
        counter += 1
        plt.close('all')

    return res['x']


run(lambda x, y: x ** 2 - 2 * x * y + y ** 2 + 1, -100, 50)
# print(history)


