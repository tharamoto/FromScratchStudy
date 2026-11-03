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
from dezero.utils import plot_dot_graph

""" 
x = Variable(np.random.randn(2,3))
x.name = 'x'
print(_dot_var(x))
print(_dot_var(x, verbose=True))
"""

""" x1 = Variable(np.array(1.0))
x0 = Variable(np.array(1.0))
y = x0 + x1
txt = _dot_func(y.creator)
print(txt) """

# def goldstein(x, y):
#     z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)) * \
#     (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))

#     return z

# x = Variable(np.array(1.0))
# y = Variable(np.array(1.0))
# z = goldstein(x, y)
# z.backward()

# x.name = 'x'
# y.name = 'y'
# z.name = 'z'

# plot_dot_graph(z, verbose=True, to_file='goldstein.png')


from pathlib import Path
import subprocess

repo = "/Users/haramototakeshi/FromScratch/FromScratchStudy_filter_work"
mapping = {}

for line in Path("/Users/haramototakeshi/FromScratch/commit_message_map.txt").read_text(
    encoding="utf-8"
).splitlines():
    short_id, new_message = line.split("|", 1)

    full_id = subprocess.check_output(
        [
            "git",
            "-C",
            repo,
            "rev-parse",
            "--verify",
            short_id
        ],
        text=True
    ).strip()

    if full_id in mapping:
        raise ValueError(f"Duplicate commit: {short_id}")

    mapping[full_id] = new_message

assert len(mapping) == 30

output = Path("/Users/haramototakeshi/FromScratch/commit_message_map.txt")
output.write_text(
    "".join(
        f"{commit_id}|{message}\n"
        for commit_id, message in mapping.items()
    ),
    encoding="utf-8"
)

print(f"Created {output}: {len(mapping)} entries")