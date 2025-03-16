import numpy as np

def dec2bin(x, bits):
    mask = 2 ** np.arange(bits - 1, -1, -1, dtype=x.dtype)
    return np.array(
        np.not_equal(np.bitwise_and(np.expand_dims(x, -1), mask), 0), dtype="long"
    )


def bin2dec(b, bits):
    mask = 2 ** np.arange(bits - 1, -1, -1, dtype=b.dtype)
    return np.sum(mask * b, -1)


def transform_idxs(x, bits):
    if not isinstance(x, np.ndarray):
        x = np.stack(x)
    if x.dtype != np.uint64:
        x = np.array(x, dtype="long")
    bits_transformed = x.shape[0]
    x_bin = dec2bin(x.swapaxes(0, -1), bits).swapaxes(-2, -1)
    x_transformed = bin2dec(x_bin, bits_transformed).swapaxes(0, -1)
    return tuple(x_transformed)


def transform_tensor(tensor, resolution=None):
    ndim = tensor.ndim
    exp = int(np.log2(tensor.shape[0]))
    if resolution == None:
        current_idx = np.arange(0, 2**ndim)
    else:
        current_idx = np.arange(0, 2**ndim, max(1, 2 ** (exp - resolution)))
    current_idxs = np.meshgrid(*([current_idx] * exp), indexing="ij")
    transformed_idxs = transform_idxs(current_idxs, ndim)
    return tensor[transformed_idxs]


def transform_tensors(tensors):
    return [transform_tensor(tensor) for tensor in tensors]