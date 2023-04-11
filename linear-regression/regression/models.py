'''
Various models for the experiments
'''

from regression.bases import *
from regression.dataset import *
from regression.loss import *
from regression.model import *

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


def make_simple_model():
    '''
    loss: MSE
    bases: all polynomials of 7 variables with max degree 7
    '''
    polys = []
    for p in itertools.combinations_with_replacement(range(-1, 7), 7):
        vars = [i for i in p if i >= 0]
        poly = MutltiVarPolynomial(vars)
        polys.append(poly)
    return Model(MSE(), polys)


def make_simple_model_l2(l2):
    model = make_simple_model()
    model.loss = Sum(MSE(), L2(l2))
    return model


def make_simple_model_l1(l1):
    model = make_simple_model()
    model.loss = Sum(MSE(), L1(l1))
    return model


def make_simple_model_elastic_net(l1, l2):
    model = make_simple_model()
    model.loss = Sum(MSE(), L1(l1), L2(l2))
    return model
