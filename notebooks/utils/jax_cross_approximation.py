import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import opt_einsum as oe

# import numpy as np
from utils.jax_ops import *


def cross(
    random_key,
    f,
    # domain=None,
    variables=None,  # list of variables in mps form [x, y, ...]
    # function_arg="vectors",
    bond_dims=None,
    kickrank=3,
    rmax=100,
    eps=1e-6,
    max_iter=25,
    val_size=1000,
    verbose=True,
    return_info=False,
):

    # if bonds are not specified, match the last variable
    if bond_dims == None:
        bond_dims = bond_dims_of_MPS(variables[-1])
        shapes = [t.shape for t in variables[-1]]
    else:
        phys_dims = phys_dims_of_MPS(variables[-1])
        shapes = [
            (bond_dims[i], phys_dims[i], bond_dims[i + 1])
            for i in range(len(phys_dims))
        ]

    # initialize random cores
    subkeys = []
    for i in range(len(bond_dims) - 1):
        random_key, subkey = jax.random.split(random_key)
        subkeys += [subkey]
    cores = [jax.random.uniform(subkeys[i], shapes[i]) for i in range(len(shapes))]

    # Prepare left and right sets
    lsets = [jnp.array([[0]])] + [None] * (N - 1)
    randint = jnp.hstack(
        [jnp.random.randint(0, Is[n + 1], [max(Rs), 1]) for n in range(N - 1)]
        + [jnp.zeros([max(Rs), 1])]
    )
    rsets = [randint[: Rs[n + 1], n:] for n in range(N - 1)] + [np.array([[0]])]
    return cores
