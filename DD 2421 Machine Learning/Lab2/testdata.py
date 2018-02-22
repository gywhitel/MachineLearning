import numpy

numpy.random.seed(100)

classA = numpy.concatenate(
    (numpy.random.randn(10, 2) * 0.2 + [1.5, 0.5],
     numpy.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))

classB = numpy.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = numpy.concatenate((classA, classB))
targets = numpy.concatenate((numpy.ones(classA.shape[0]),
                             -numpy.ones(classB.shape[0])))

N = inputs.shape[0] # Number of rows

permute = list(range(N))
numpy.random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]


print(inputs)
print(len(inputs))
