import re

import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize


rosen = (lambda x: 100.0 * (x[0] - x[1]**2.0)**2.0 + (1 - x[1])**2.0)


x0 = np.array([1.3 * 5, 0.7 * 5])

res = minimize(rosen, x0, method='Newton-CG', jac=(lambda x: opt.approx_fprime(x, rosen)))

print(f'Newton-CG: x={res.x}, iters={res.nit}, status: {res.message}')

res = minimize(rosen, x0, method='BFGS')

print(f'BFGS: x={res.x}, iters={res.nit}, status: {res.message}')
