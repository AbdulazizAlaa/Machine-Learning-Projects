import theano.tensor as T
from theano import function

a = T.vector()
b = T.vector()

out = a ** 2 + b ** 2 + 2 * a * b

f = function((a,b), out)

print f([1, 2], [4, 5])
