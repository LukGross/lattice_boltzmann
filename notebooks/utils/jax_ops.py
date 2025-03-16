import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as onp

import opt_einsum as oe


def list_to_jnp(list_of_nps):
    if isinstance(list_of_nps[0], list):
        return [list_to_jnp(entry) for entry in list_of_nps]
    else:
        return [jnp.array(entry) for entry in list_of_nps]


def list_to_np(list_of_jnps):
    if isinstance(list_of_jnps[0], list):
        return [list_to_np(entry) for entry in list_of_jnps]
    else:
        return [onp.array(entry) for entry in list_of_jnps]


def rq(matrix):
    Q, R = jnp.linalg.qr(matrix.T)
    return R.T, Q.T


def qr(matrix):
    return jnp.linalg.qr(matrix)


def truncated_svd(matrix, max_bond=None):
    U, S, V = jnp.linalg.svd(matrix, full_matrices=False)
    if not max_bond == None:
        cutoff = min(max_bond, len(S))
        U = U[:, :cutoff]
        S = S[:cutoff]
        V = V[:cutoff, :]
    return U, S, V


def right_shift_canonical_center(tensors, current_center):
    tensors = tensors.copy()
    right = current_center + 1
    current_shape = tensors[current_center].shape
    right_shape = tensors[right].shape
    bond_dim = current_shape[-1]
    matrix = tensors[current_center].reshape(-1, bond_dim)
    Q, R = qr(matrix)
    tensors[current_center] = Q.reshape(current_shape[:-1] + (Q.shape[-1],))
    tensors[right] = jnp.tensordot(R, tensors[right], axes=1)
    return tensors


def left_shift_canonical_center(tensors, current_center):
    tensors = tensors.copy()
    left = current_center - 1
    current_shape = tensors[current_center].shape
    left_shape = tensors[left].shape
    bond_dim = current_shape[0]
    matrix = tensors[current_center].reshape(bond_dim, -1)
    R, Q = rq(matrix)
    tensors[current_center] = Q.reshape((Q.shape[0],) + current_shape[1:])
    tensors[left] = jnp.tensordot(tensors[left], R, axes=1)
    return tensors


def canonicalize(tensors, center, current_center=None):
    tensors = tensors.copy()
    for i in range(center):
        tensors = right_shift_canonical_center(tensors, i)
    for i in range(len(tensors) - 1, center, -1):
        tensors = left_shift_canonical_center(tensors, i)
    return tensors


def compress_MPS(tensors, max_bond=None):
    tensors = tensors.copy()
    tensors = canonicalize(tensors, len(tensors) - 1)
    for i in range(len(tensors) - 1, 0, -1):
        shape_left, shape_phys, shape_right = tensors[i].shape
        U, S, V = truncated_svd(
            tensors[i].reshape(shape_left, shape_phys * shape_right), max_bond
        )
        tensors[i] = V.reshape(-1, shape_phys, shape_right)
        tensors[i - 1] = jnp.einsum("abc, cd -> abd", tensors[i - 1], U @ jnp.diag(S))
    return tensors


def MPS(tensor, max_bond=None):
    shape = tensor.shape
    MPS_tensors = []
    shape_right = 1
    for site in range(len(shape) - 1, 0, -1):
        shape_phys = shape[site]
        U, S, V = truncated_svd(tensor.reshape(-1, shape_phys * shape_right), max_bond)
        tensor = U @ jnp.diag(S)
        shape_left = len(S)
        MPS_tensors = [V.reshape(shape_left, shape_phys, shape_right)] + MPS_tensors
        shape_right = shape_left
    MPS_tensors = [tensor.reshape(1, shape_phys, shape_right)] + MPS_tensors
    return MPS_tensors


def bond_dims_of_MPS(tensors):
    bond_dims = (tensors[0].shape[0],)
    for t in tensors:
        bond_dims += (t.shape[-1],)
    return bond_dims


def phys_dims_of_MPS(tensors):
    phys_dims = ()
    for t in tensors:
        phys_dims += (t.shape[1],)
    return phys_dims


def bond_dims_of_MPO(tensors):
    bond_dims = (tensors[0].shape[0],)
    for t in tensors:
        bond_dims += (t.shape[-1],)
    return bond_dims


def phys_dims_of_MPO(tensors):
    phys_dims = ()
    for t in tensors:
        phys_dims += (t.shape[1],)
    return phys_dims


def contract_MPS(tensors):
    ein_str = ""
    out_str = ""
    offs = ord("a")
    for i in range(len(tensors) - 1):
        ein_str += chr(2 * i + offs)
        ein_str += chr(2 * i + 1 + offs)
        out_str += chr(2 * i + 1 + offs)
        ein_str += chr(2 * i + 2 + offs)
        ein_str += ","
    i = len(tensors) - 1
    ein_str += chr(2 * i + offs)
    ein_str += chr(2 * i + 1 + offs)
    out_str += chr(2 * i + 1 + offs)
    ein_str += "a"
    ein_str = ein_str + "->" + out_str
    return oe.contract(ein_str, *tensors)


def apply_MPO(mpo, mps):
    out = []
    for i in range(len(mpo)):
        shape_left = mpo[i].shape[0] * mps[i].shape[0]
        shape_phys = mpo[i].shape[1]
        shape_right = mpo[i].shape[3] * mps[i].shape[2]
        out += [
            jnp.einsum("abcd, ecg -> aebdg", mpo[i], mps[i]).reshape(
                shape_left, shape_phys, shape_right
            )
        ]
    return out


def apply_MPO_MPO(mpo2, mpo1):
    out = []
    for i in range(len(mpo2)):
        shape_left = mpo2[i].shape[0] * mpo1[i].shape[0]
        shape_out = mpo2[i].shape[1]
        shape_in = mpo1[i].shape[2]
        shape_right = mpo2[i].shape[3] * mpo1[i].shape[3]
        out += [
            jnp.einsum("abcd, ecgh -> aebgdh", mpo2[i], mpo1[i]).reshape(
                shape_left, shape_out, shape_in, shape_right
            )
        ]
    return out


def multiply_MPS_MPS(kron, mps1, mps2):
    out = []
    for i in range(len(mps1)):
        shape_left = mps1[i].shape[0] * mps2[i].shape[0]
        shape_phys = kron.shape[0]
        shape_right = mps1[i].shape[2] * mps2[i].shape[2]
        out += [
            oe.contract("abc, dbf, gci -> dgafi", kron, mps1[i], mps2[i]).reshape(
                shape_left, shape_phys, shape_right
            )
        ]
    return out


def multiply_MPO_MPO(kron, mpo1, mpo2):
    out = []
    for i in range(len(mpo1)):
        shape_left = mpo1[i].shape[0] * mpo2[i].shape[0]
        shape_out = kron.shape[0]
        shape_in = shape_out
        shape_right = mpo1[i].shape[-1] * mpo2[i].shape[-1]
        out += [
            oe.contract(
                "abc, dbfg, hcjk, fjn -> dhangk", kron, mpo1[i], mpo2[i], kron
            ).reshape(shape_left, shape_out, shape_out, shape_right)
        ]
    return out


def add_MPS_MPS(mps1, mps2):
    out = [
        jnp.zeros(
            (
                1,
                mps1[0].shape[1],
                mps1[0].shape[2] + mps2[0].shape[2],
            )
        )
    ]
    out[0] = out[0].at[:, :, : mps1[0].shape[2]].set(mps1[0])
    out[0] = out[0].at[:, :, mps1[0].shape[2] :].set(mps2[0])

    for i in range(1, len(mps1) - 1):
        out += [
            jnp.zeros(
                (
                    mps1[i].shape[0] + mps2[i].shape[0],
                    mps1[i].shape[1],
                    mps1[i].shape[2] + mps2[i].shape[2],
                )
            )
        ]
        out[-1] = out[-1].at[: mps1[i].shape[0], :, : mps1[i].shape[2]].set(mps1[i])
        out[-1] = out[-1].at[mps1[i].shape[0] :, :, mps1[i].shape[2] :].set(mps2[i])
    out += [
        jnp.zeros(
            (
                mps1[-1].shape[0] + mps2[-1].shape[0],
                mps1[-1].shape[1],
                1,
            )
        )
    ]
    out[-1] = out[-1].at[: mps1[-1].shape[0], :, :].set(mps1[-1])
    out[-1] = out[-1].at[mps1[-1].shape[0] :, :, :].set(mps2[-1])
    return out
