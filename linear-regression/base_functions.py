'''
Collection of base functions
'''

import numpy as np


def projection(i):
    return lambda x: x[i]


def projections(n):
    return [projection(i) for i in range(n)]


def polynomial(d):
    '''
    Polynomial of degree d
    '''
    return lambda x: x ** d


def polynomials(n):
    '''
    Polynomials of degree 1 to n
    '''
    return [polynomial(i) for i in range(1, n + 1)]


def const(c):
    return lambda _: c


def mul(f, g):
    return lambda x: f(x) * g(x)


def compose(f, g):
    return lambda x: g(f(x))


def exp():
    return lambda x: np.exp(x)
