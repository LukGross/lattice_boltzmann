import numpy as np
import findiff


def Id(bond, d=2):
    if isinstance(d, int):
        return np.tensordot(
            np.eye(bond, dtype=float), np.eye(d, dtype=float), axes=0
        ).transpose(0, 2, 3, 1)


def extend_to_w(nucleus, axis, physical_dimensions):
    core = nucleus
    left = core.shape[0]
    right = core.shape[-1]
    for i in range(axis):
        core = np.einsum("abcd,dklm->abkclm", Id(left), core)
        output = core.shape[1] * core.shape[2]
        input = core.shape[3] * core.shape[4]
        core = core.reshape(left, output, input, right)
    for i in range(axis + 1, physical_dimensions):
        core = np.einsum("abcd,dklm->abkclm", core, Id(right))
        output = core.shape[1] * core.shape[2]
        input = core.shape[3] * core.shape[4]
        core = core.reshape(left, output, input, right)
    return core


def toeplitz4_cores(coeffs, resolution_exponent, boundary="periodic"):
    l = len(coeffs)
    if not l == 9:
        raise Exception("all 9 coefficients need to be specified")
    # scale = max(abs(coeff) for coeff in coeffs)
    # coeffs = [coeff / scale for coeff in coeffs]
    a, b, c, d, e, i, h, g, f = coeffs
    I = np.identity(2)
    J = np.array([[0, 1], [0, 0]])
    J_ = J.T
    P = J + J_
    Z = np.zeros((2, 2))

    if boundary == "periodic":
        A = np.array([[I, P, P]]).transpose(0, 2, 3, 1)
    if boundary == "fixed":
        A = np.array([[I, J_, J]]).transpose(0, 2, 3, 1)
    if boundary == "open":
        A = "good question!"  # TODO
        raise Exception("i dunno")
    B = np.array([[I, J_, J], [Z, J, Z], [Z, Z, J_]]).transpose(0, 2, 3, 1)
    C = np.array([[I, J_, J, Z, Z], [Z, J, Z, I, Z], [Z, Z, J_, Z, I]]).transpose(
        0, 2, 3, 1
    )
    D = np.array(
        [
            [a * I + b * J + f * J_],
            [g * I + f * J + h * J_],
            [c * I + d * J + b * J_],
            [i * I + h * J + 0.0 * J_],
            [e * I + 0.0 * J + d * J_],
        ]
    ).transpose(0, 2, 3, 1)
    cores = [A] + [B] * (resolution_exponent - 3) + [C] + [D]
    # cores = [(scale ** (1 / resolution_exponent)) * core for core in cores]
    return cores


def diff_cores(
    diff_order,
    axis,
    resolution_exponent,
    physical_dimensions,
    acc_order=8,
    delta_x=1,
    boundary="periodic",
):
    if diff_order < 0:
        raise Exception("diff_order cant be negative")
    if (diff_order + acc_order) / 2 > 5:
        raise Exception("order not implemented")
    coeffs = findiff.coefficients(diff_order, acc_order, symbolic=True)["center"][
        "coefficients"
    ]
    coeffs = np.array(coeffs)
    coeffs = coeffs / delta_x**diff_order
    pad = 4 - int(acc_order / 2) - int((diff_order - 1) / 2)
    coeffs = np.roll(np.pad(coeffs, pad), -4).astype(np.float64).tolist()
    nuclei = toeplitz4_cores(coeffs, resolution_exponent, boundary)
    cores = [extend_to_w(nucleus, axis, physical_dimensions) for nucleus in nuclei]
    return cores


def kron_delta(dim, rank=3):
    delt = np.zeros([dim] * rank)
    for i in range(dim):
        delt[(i,) * rank] = 1
    return delt


def kron_cores(sim_params):
    resolution_exponent = sim_params["resolution_exponent"]
    physical_dimensions = sim_params["physical_dimensions"]
    return [kron_delta(2**physical_dimensions)] * resolution_exponent


def gen_d_list(diff_order, sim_params):
    resolution_exponent = sim_params["resolution_exponent"]
    physical_dimensions = sim_params["physical_dimensions"]
    delta_x = sim_params["delta_x"]
    return [
        diff_cores(
            diff_order=diff_order,
            axis=k,
            resolution_exponent=resolution_exponent,
            physical_dimensions=physical_dimensions,
            delta_x=delta_x,
        )
        for k in range(physical_dimensions)
    ]
