import os
import subprocess
import numpy as np


def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orage, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        name += ': '

    name += str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name)


