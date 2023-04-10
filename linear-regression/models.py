'''
Various models for the experiments
'''

from bases import *
from dataset import *
from loss import *
from model import *

import itertools

FEATURES = 7
PROJECTIONS = projections(FEATURES)


def make_simplest_model():
    '''
    loss: MSE
    bases: all polynomials of 7 variables with max degree 3
    '''
    polys = []
    for p in itertools.combinations_with_replacement(range(-1, 7), 3):
        vars = [i for i in p if i >= 0]
        poly = MutltiVarPolynomial(vars)
        polys.append(poly)
    return Model(MSE(), polys)
