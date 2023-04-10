'''
Collection of base functions
'''

import numpy as np


class Projection:
    def __init__(self, i):
        self.i = i

    def __call__(self, x):
        return x[self.i]

    def __repr__(self):
        return f"x[{self.i}]"


def projections(n):
    return [Projection(i) for i in range(n)]


class Polynomial:
    def __init__(self, d):
        self.d = d

    def __call__(self, x):
        return x ** self.d

    def __repr__(self):
        return f"x ** {self.d}"


class MutltiVarPolynomial:
    def __init__(self, vars):
        self.vars = vars

    def __call__(self, x):
        return np.prod([x[i] for i in self.vars])

    def __repr__(self):
        result = " * ".join([f"x[{i}]**{self.vars.count(i)}" for i in set(self.vars)])
        return result or "1"


def polynomials(n):
    '''
    Polynomials of degree 1 to n
    '''
    return [Polynomial(i) for i in range(1, n + 1)]


class Const:
    def __init__(self, c):
        self.c = c

    def __call__(self, _):
        return self.c

    def __repr__(self):
        return f"{self.c}"


class Mul:
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __call__(self, x):
        return self.f(x) * self.g(x)

    def __repr__(self):
        return f"({self.f} * {self.g})"


class Add:
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __call__(self, x):
        return self.f(x) + self.g(x)

    def __repr__(self):
        return f"({self.f} + {self.g})"


class Composition:
    def __init__(self, f, g):
        self.f = f
        self.g = g

    def __call__(self, x):
        return self.g(self.f(x))

    def __repr__(self):
        return f"({self.g} . {self.f})"


class Exp:
    def __init__(self, a):
        self.a = a

    def __call__(self, x):
        return np.exp(self.a * x)

    def __repr__(self):
        return f"exp({self.a} * x)"


class Normal:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x):
        return np.exp(-((x - self.mu) ** 2) / (2 * self.sigma ** 2))

    def __repr__(self):
        return f"exp(-((x - {self.mu}) ** 2) / (2 * {self.sigma} ** 2))"


class Mean:
    def __call__(self, x):
        return np.mean(x)

    def __repr__(self):
        return f"mean(x)"
