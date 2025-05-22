import torch
import cupy
import numpy as np
from torch.utils.dlpack import to_dlpack, from_dlpack
from scipy.linalg import hadamard

def generate_hadamard_matrix(size):
    """
    generate hadamard matrix
    for 2^n x 2^n matrix, use sylvester construction
    """
    if size & (size - 1) == 0:  # size must be a power of 2
        H = torch.from_numpy(hadamard(size)).to(torch.float32)
        return H.cuda()
    else:
        raise ValueError("hadamard matrix size must be a power of 2")

def cupy_to_tensor(x):
    return from_dlpack(x.toDlpack())

def tensor_to_cupy(x):
    return cupy.fromDlpack(to_dlpack(x))

def pack_low_bit_tensor(x, bits):
    assert x.dtype == torch.uint8
    y = cupy.packbits(
        cupy.unpackbits(tensor_to_cupy(x)).reshape(*x.shape, 8)[..., -bits:]
    )
    y = cupy_to_tensor(y)
    return y

def unpack_low_bit_tensor(x, bits, original_shape):
    y = cupy.packbits(cupy.pad(
        cupy.unpackbits(
            tensor_to_cupy(x)
        )[:np.prod(original_shape)*bits].reshape(-1, bits),
        ((0,0), (8-bits, 0))
    ))
    y = cupy_to_tensor(y).view(original_shape)
    return y


def pin_memory(array):
    mem = cupy.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret