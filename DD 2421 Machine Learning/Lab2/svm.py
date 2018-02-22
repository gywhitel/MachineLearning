import numpy, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import testdata as td

random.seed(100)

# Parameters
# N = number of training samples
# C = biggest value possible, 0 is smallest
# XC = second constraint from (10)
# P = N x N matrix
N = 40
C = 2
x = td.inputs
t = td.targets
P = numpy.zeros((N, N))
threshold = 1e-5


def objective(a):
    """

    :param a: vector
    :return: the scalar value in the dual form
    """
    # ( 1/2 Sum i Sum j ai aj Pij ) - Sum
    cumsum = 0
    for i in range(len(a)):
        for j in range(len(a)):
            cumsum += numpy.dot(numpy.dot(a[i], a[j]), P[i][j])

    retval = 0.5 * cumsum - numpy.sum(a)

    # inner = numpy.dot(a, P)
    return retval  # 0.5 * numpy.sum(inner) - numpy.sum(a)


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

for i in range(0, N):
    for j in range(0, N):
        P[i][j] = (t[i] * t[j] * kernel_linear(x[i], x[j]))

ret = minimize(objective, start(N), bounds=B, constraints=XC)
alpha = ret['x']
done = ret['success']

new_x = []
new_t = []
new_a = []

for i in range(len(alpha)):
    if alpha[i] < threshold:
        alpha[i] = 0
    else:
        new_x.append(x[i])
        new_t.append(t[i])
        new_a.append(alpha[i])

print(new_t)
print(new_x)
print(done)
print(alpha)

b = 0
for i in range(len(x)):
    sv = random.randint(0, len(new_x) - 1)
    b += alpha[i] * t[i] * kernel_linear(new_x[sv], x[i]) - new_t[sv]

print(b)


def indicator(xx, yy):
    su = 0
    for i in range(len(new_x)):
        su += new_a[i] * new_t[i] * kernel_linear(numpy.array([xx, yy]), new_x[i])
    return su - b


plt.plot([p[0] for p in td.classA],
         [p[1] for p in td.classA],
         'b.')
plt.plot([p[0] for p in td.classB],
         [p[1] for p in td.classB],
         'r.')


xgrid = numpy.linspace(-5, 5)
ygrid = numpy.linspace(-4, 4)

grid = numpy.array([[indicator(x, y) for y in ygrid] for x in xgrid])

print(grid)


plt.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))

plt.axis('equal')
plt.savefig('svmplot.pdf')
plt.show()
