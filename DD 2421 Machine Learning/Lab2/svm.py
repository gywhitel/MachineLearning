import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import testdata as td

# Parameters
# N = number of training samples
# C = biggest value possible, 0 is smallest
# XC = second constraint from (10)
# P = N x N matrix
N = 40
C = None
x = td.inputs
t = td.targets
P = numpy.zeros((N, N))


def objective(a):
    """

    :param a: vector
    :return: the scalar value in the dual form
    """
    # ( 1/2 Sum i Sum j ai aj Pij ) - Sum
    inner = numpy.dot(a, P)
    0.5 * numpy.sum(inner) - numpy.sum(a)
    return


def start(N):
    """
    :return: give a vector an initial value
    """
    return numpy.zeros(N)


def kernel_linear(x, y):
    """
    :param x: vector
    :param y: vector
    :return: scalar product of x transposed multiplied with y
    """
    return numpy.dot(x, y)


def kernel_polynomial(x, y, p):
    """

    :param x: vector
    :param y: vector
    :param p: degree
    :return:
    """
    return (numpy.dot(x, y) + 1) ** p


def kernel_radial(x, y, o):
    """

    :param x: vector
    :param y: vector
    :param o: sigma
    :return:
    """
    return math.e ** (-(numpy.transpose(x - y) * (x - y) ** 2) / (2 * o ** 2))


B = [(0, C) for b in range(N)]


def zerofun(a):
    """
    equality constraint
    :return: scalar value
    """
    return numpy.sum(numpy.dot(a, t))


XC = {'type': 'eq', 'fun': zerofun}


def main():
    for i in range(0, N):
        for j in range(0, N):
            P[i][j] = (t[i] * t[j] * kernel_linear(x[i], x[j]))

    ret = minimize(objective, start(N), bounds=B, constraints=XC)
    alpha = ret['x']
    print(alpha)

main()
