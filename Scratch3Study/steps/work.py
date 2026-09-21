# import numpy as np
# from dezero.core_simple import Variable

# x = Variable(np.array(1.0))
# print(x)


if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

"""
import numpy as np
from dezero import Variable

x = Variable(np.array(1.0))
y = (x + 3) ** 2
y.backward()

print(y)
print(x.grad)
 """

import numpy as np
from dezero import Variable
from dezero.utils import _dot_var
from dezero.utils import _dot_func


""" 
x = Variable(np.random.randn(2,3))
x.name = 'x'
print(_dot_var(x))
print(_dot_var(x, verbose=True))
"""

x1 = Variable(np.array(1.0))
x0 = Variable(np.array(1.0))
y = x0 + x1
txt = _dot_func(y.creator)
print(txt)