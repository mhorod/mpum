'''
Various models for the experiments
'''

from bases import *
from dataset import *
from loss import *
from model import *


FEATURES = 7
PROJECTIONS = projections(FEATURES)


def make_simplest_model():
    '''
    loss: MSE
    bases: all polynomials of max degree 3
    '''
    polys = polynomials(3)
    polys = [Composition(proj, poly) for proj in PROJECTIONS for poly in polys]
    polys += [Mul(f, g) for f in polys for g in polys]
    base_functions = [Const(1)] + polys
    return Model(MSE(), base_functions)
