
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import cupy
from torch.utils.dlpack import to_dlpack, from_dlpack
import concurrent.futures
import tempfile
import time

from .fixpoint import *
from .utils import *
from . import flag

import math
from math import log2

class TAHQuantCompressor:
    def __init__(
            self, 
            high_precision_bits=4,
            low_precision_bits=3,
            scale_method='max',
            tile_size=64,
            high_precision_allocation_ratio=0.8,
            *args, **kargs,
    ):
        self.high_precision_bits = high_precision_bits
        self.low_precision_bits = low_precision_bits
        self.scale_method = scale_method
        self.tile_size = tile_size
        self.high_precision_allocation_ratio = high_precision_allocation_ratio
        assert tile_size & (tile_size - 1) == 0, "tile_size must be a power of 2"
        assert high_precision_bits <= 8, "high_precision_bits for THAQuantCompressor must be less than or equal to 8"
        assert low_precision_bits <= 8, "low_precision_bits for THAQuantCompressor must be less than or equal to 8"

    def build_buffer(self, batch_size, micro_batch_size, seq_length, embedding_dim, device, dtype=None):
        self.micro_batch_size = micro_batch_size
        self.activ_shape = (micro_batch_size, seq_length, embedding_dim)
        self.device = device
        self.dtype = dtype

        # Communication Buffers
        B, S, C = self.activ_shape
        assert C % self.tile_size == 0, "C must be divisible by tile_size"
        """
        quantizatized activation.
        --- high precision part ---
        bit-width:
            self.high_precision_bits
        element-number:
            B * int(S * high_precision_allocation_ratio) * C
        total-bit-num:
            high_precision_bits * element_number
        --- low precision part ---
        bit-width:
            self.low_precision_bits
        element-number:
            B * int(S * (1 - high_precision_allocation_ratio)) * C
        total-bit-num:
            low_precision_bits * element_number
        """
        activation_bit_num = B * C * (
            int(S * self.high_precision_allocation_ratio) * self.high_precision_bits + 
            int(S * (1 - self.high_precision_allocation_ratio)) * self.low_precision_bits
        )
        """
        zero point.
        bit-width:
            fp16: 16-bit
        element-number:
            B * S * C // self.tile_size
        total-bit-num:
            B * S * C // self.tile_size * 16
        """
        zero_point_bit_num = B * S * C // self.tile_size * 16
        """
        scale.
        bit-width:
            fp16: 16-bit
        element-number:
            B * S * C // self.tile_size
        total-bit-num:
            B * S * C // self.tile_size * 16
        """
        scale_bit_num = B * S * C // self.tile_size * 16
        """
        token_bit_map.
        bit-width:
            1-bit. 1 indicates the token is allocated with high precision, 0 indicates low precision.
        element-number:
            B * S
        total-bit-num:
            B * S
        """
        token_bit_map_bit_num = B * S
        """
        tile_bit_map.
        bit-width:
            1-bit. 1 indicates the tile needs to do the Pivot Swap and Hadamard Transformation.
        element-number:
            B * S * C // self.tile_size
        total-bit-num:
            B * S * C // self.tile_size
        """
        tile_bit_map_bit_num = B * S * C // self.tile_size
        """
        pivot_indices.
        bit-width:
            int(log2(self.tile_size))
        element-number:
            number of 1 in tile_bit_map
        total-bit-bum:
            number of 1 in tile_bit_map * int(log2(self.tile_size))
        max-bit-num:
            int(log2(self.tile_size)) * B * S * C // self.tile_size
        """
        
        pivot_indices_max_bit_num = math.ceil((log2(self.tile_size))) * B * S * C // self.tile_size
        total_bit_num = activation_bit_num + zero_point_bit_num + scale_bit_num + token_bit_map_bit_num + tile_bit_map_bit_num + pivot_indices_max_bit_num
        total_byte_num = int(total_bit_num / 8)
        self.buffers = [
            torch.zeros(total_byte_num, requires_grad=False, device=device, dtype=torch.uint8) for _ in range(batch_size//micro_batch_size)
        ]
        
        # Communication Buffers during Warmup (w/o compression)
        self.warmup_buffers = [
            torch.zeros((micro_batch_size, seq_length, embedding_dim), 
                        requires_grad=False, device=device, dtype=dtype,
                       ) for _ in range(batch_size//micro_batch_size)
        ]

        self.H = generate_hadamard_matrix(self.tile_size)
        self.H = self.H.to(device=device, dtype=dtype)

    def compress(self, x):
        B, S, C = self.activ_shape
        device = x.device
        dtype = x.dtype
        """
        token-level entropy-based precision allocation.
        token_bit_map: (B, S), uint8, effective bit for each is 1
        """
        x_abs = torch.abs(x)
        # normalized
        eps = 1e-30
        probs = x_abs / (x_abs.sum(dim=-1, keepdim=True) + 1e-30) # (B, S, C)
        # compute entropy
        entropy = -(probs * (probs + eps).log()).sum(dim=-1) # (B, S)
        entropy_based_idxs = torch.argsort(entropy, dim=-1, descending=False) # (B, S)
        # B_idxs = torch.arange(B).unsqueeze(1).expand(-1, S)
        B_idxs = torch.arange(B).unsqueeze(1)
        low_prec_seq_len = int(S * (1 - self.high_precision_allocation_ratio))
        low_prec_idxs = entropy_based_idxs[:, :low_prec_seq_len]
        high_prec_idxs = entropy_based_idxs[:, low_prec_seq_len:]
        low_prec_idxs_sort, _ = torch.sort(low_prec_idxs, dim=1)
        high_prec_idxs_sort, _ = torch.sort(high_prec_idxs, dim=1)
        S_idxs_sort = torch.cat([low_prec_idxs_sort, high_prec_idxs_sort], dim=1)
        x = x[B_idxs, S_idxs_sort]
        # need send info of S_indxs_sort to next stage for decompression
        token_bit_map = torch.zeros((B, S), device=device, dtype=torch.uint8)
        token_bit_map[B_idxs, low_prec_idxs] = 1 # (B, S), effective bit for each is 1

        """
        determine if the tile needs to do the Pivot Swap and Hadamard Transformation.
        tile_bit_map: (B * S * C // self.tile_size), uint8, effective bit for each is 1
        """
        x = x.reshape(B * S * C // self.tile_size, self.tile_size) # (num_tiles,)
        top2_abs_valw, _ = torch.topk(torch.abs(x), k=2, dim=1, largest=
        True, sorted=True)
        first, second = top2_abs_valw[:, 0], top2_abs_valw[:, 1]
        apply_mask = (first >= 2 * second) # (num_tiles,)
        tile_bit_map = apply_mask.to(torch.uint8)

        """
        Pivod swap and Hadamard Transformation.
        pivot_indices: (apply_num_tiles, ), int64, effective bit log2(tile_size)
        """
        apply_indices = torch.nonzero(apply_mask,).squeeze(1).to(device=device)
        pivot_indices = None
        if apply_indices.numel() > 0:
            selected_tiles = x[apply_indices] # (apply_num_tiles, tile_size)
            pivot_indices = torch.abs(selected_tiles).argmax(dim=1) # (apply_num_tiles, )
            rows = torch.arange(apply_indices.shape[0], device=device)
            cols = pivot_indices
            # pivot swap
            first_vals = selected_tiles[:, 0].clone()
            abs_max_vals = selected_tiles[rows, cols].clone()
            selected_tiles[rows, cols] = first_vals
            selected_tiles[:, 0] = abs_max_vals
            # Hadamard Transformation
            selected_tiles = torch.matmul(selected_tiles, self.H) / torch.sqrt(torch.tensor(self.tile_size, dtype=dtype, device=device))
            x[apply_indices] = selected_tiles

        """
        zero point shift.
        zeros: (B, S, C // tile_size, 1), x.dtype, effective bit for each is 16(fp16)
        """
        x = x.reshape(B, S, C // self.tile_size, self.tile_size)
        zeros = torch.min(x, dim=3, keepdim=True)[0] # (B, S, C // self.tile_size, 1)
        x -= zeros
        if (x < 0).any():
            raise ValueError("x with zero point shift has negative values")

        """
        quantize.
        scales: (B, S, C // tile_size, 1), x.dtype, effective bit for each is 16(fp16)
        """
        scales = torch.max(torch.abs(x), dim=3, keepdim=True)[0] # (B, S, C // tile_size, 1)
        low_prec_seq_len = int(S * (1 - self.high_precision_allocation_ratio))
        S1 = low_prec_seq_len
        LPM = (2**self.low_precision_bits) - 1
        HPM = (2**self.high_precision_bits) - 1
        scales[:, :S1, :] = scales[:, :S1, :] / LPM
        scales[:, S1:, :] = scales[:, S1:, :] / HPM
        eps = 1e-30
        scales = torch.maximum(scales, torch.tensor(eps, dtype=dtype, device=device))
        x /= scales
        x[:, :S1, :] = torch.clamp(torch.round(x[:, :S1, :]), 0, LPM)
        x[:, S1:, :] = torch.clamp(torch.round(x[:, S1:, :]), 0, HPM)
        x = x.reshape(B, S, C)


        """
        x: (B, S, C), x.dtype, effective bit is low/high precision bits
        zeros: (B, S, C // tile_size, 1), x.dtype, effective bit for each is 16(fp16)
        scales: (B, S, C // tile_size, 1), x.dtype, effective bit for each is 16(fp16)
        token_bit_map: (B, S), uint8, effective bit for each is 1
        tile_bit_map: (B * S * C // tile_size), uint8, effective bit for each is 1
        pivot_indices: (apply_num_tiles, ), int64, effective bit log2(tile_size)
            apply_num_tiles = torch.nonzero(tile_bit_map).numel()
            This could be None
        """
        return x, zeros, scales, token_bit_map, tile_bit_map, pivot_indices

    def decompress(self, x, zeros, scales, token_bit_map, tile_bit_map, pivot_indices):
        B, S, C = self.activ_shape
        device = self.device
        dtype = self.dtype

        """
        x, (B, S, C // tile_size, tile_size), uint8
        zeros, (B, S, C // tile_size, 1), fp16
        scales, (B, S, C // tile_size, 1), fp16
        token_bit_map, (B, S), uint8
        tile_bit_map, (B * S * C // tile_size, ), uint8
        pivot_indices, (apply_num_tiles, ), int64
        """
        x = x.to(device=device, dtype=dtype)
        x = x * scales.to(device=device, dtype=dtype)

        """
        reverse zero point shift
        """
        x = x + zeros.to(device=device, dtype=dtype) # (B, S, C // tile_size, tile_size), fp32

        """
        reverse pivot swap and Hadamard Transformation
        """
        if pivot_indices is not None and tile_bit_map.sum() > 0:
            x = x.reshape(B * S * C // self.tile_size, self.tile_size)
            apply_indices = torch.nonzero(tile_bit_map).squeeze(1).to(device=device) # (apply_num_tiles, )
            selected_tiles = x[apply_indices] # (apply_num_tiles, tile_size)
            # reverse Hadamard Transformation
            selected_tiles = torch.matmul(selected_tiles, self.H.t()) / torch.sqrt(torch.tensor(self.tile_size, dtype=dtype, device=device))
            # reverse pivot swap
            rows = torch.arange(apply_indices.shape[0], device=device)
            cols = pivot_indices.to(torch.int64)
            first_vals = selected_tiles[:, 0].clone()
            abs_max_vals = selected_tiles[rows, cols].clone()
            selected_tiles[rows, cols] = first_vals
            selected_tiles[:, 0] = abs_max_vals
            x[apply_indices] = selected_tiles

        """
        reverse token-level entropy-based precision allocation.
        """
        x = x.reshape(B, S, C)
        # token_bit_map: (B, S), 1 represents low-precision token
        # token idxs [0, 1, ..., S-1]，for each batch
        token_indices = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)  # (B, S)
        #  bit map precision split
        low_prec_idxs = token_indices[token_bit_map == 1].reshape(B, -1)   # (B, low_prec_seq_len)
        high_prec_idxs = token_indices[token_bit_map == 0].reshape(B, -1)  # (B, high_prec_seq_len)
        # sort
        low_prec_idxs_sort, _ = torch.sort(low_prec_idxs, dim=1)
        high_prec_idxs_sort, _ = torch.sort(high_prec_idxs, dim=1)
        # cat S_idxs_sort
        S_idxs_sort = torch.cat([low_prec_idxs_sort, high_prec_idxs_sort], dim=1)  # (B, S)
        inverse_S_idxs = torch.empty_like(S_idxs_sort, device=device)
        B_idxs = torch.arange(B, device=device).unsqueeze(1).expand(-1, S)
        inverse_S_idxs[B_idxs, S_idxs_sort] = torch.arange(S, device=device).expand(B, -1)
        x = x[B_idxs, inverse_S_idxs]

        return x

    
    def pack_data(self, x, zeros, scales, token_bit_map, tile_bit_map, pivot_indices):
        """
        x: (B, S, C), x.dtype, effective bit is low/high precision bits
        zeros: (B, S, C // tile_size, 1), x.dtype, effective bit for each is 16(fp16)
        scales: (B, S, C // tile_size, 1), x.dtype, effective bit for each is 16(fp16)
        token_bit_map: (B, S), uint8, effective bit for each is 1
        tile_bit_map: (B * S * C // tile_size), uint8, effective bit for each is 1
        pivot_indices: (apply_num_tiles, ), int64, effective bit log2(tile_size)
            apply_num_tiles = torch.nonzero(tile_bit_map).numel()
            This could be None
        """
        B, S, C = self.activ_shape
        device = x.device
        dtype = x.dtype
        
        """
        pack x
        --- high precision part ---
        bit-width:
            self.high_precision_bits
        element-number:
            B * int(S * high_precision_allocation_ratio) * C
        total-bit-num:
            high_precision_bits * element_number
        --- low precision part ---
        bit-width:
            self.low_precision_bits
        element-number:
            B * int(S * (1 - high_precision_allocation_ratio)) * C
        total-bit-num:
            low_precision_bits * element_number
        """
        # low precision part
        low_prec_seq_len = int(S * (1 - self.high_precision_allocation_ratio))
        S1 = low_prec_seq_len
        target_bit = self.low_precision_bits # < 8
        assert target_bit < 8, "target_bit should be less than 8, otherwise, the data will be truncated when packing..."
        t = x[:, :S1, :].to(torch.uint8)
        t_bits = cupy.unpackbits(tensor_to_cupy(t)).reshape(*t.shape, 8)[..., -target_bit:] # (B, S1, C // tile_size, tile_size, target_bit)
        t_bits_flatten_low = t_bits.reshape(np.prod(t_bits.shape)) # (effective_bit_num, ), uint8
        # high precision part
        target_bit = self.high_precision_bits # < 8
        assert target_bit < 8, "target_bit should be less than 8, otherwise, the data will be truncated when packing..."
        t = x[:, S1:, :].to(torch.uint8)
        t_bits = cupy.unpackbits(tensor_to_cupy(t)).reshape(*t.shape, 8)[..., -target_bit:] # (B, S1, C // tile_size, tile_size, target_bit)
        t_bits_flatten_high = t_bits.reshape(np.prod(t_bits.shape)) # (effective_bit_num, ), uint8
        bits_stream = cupy.concatenate([t_bits_flatten_low, t_bits_flatten_high], axis=-1)

        """
        zeros.
        zeros: (B, S, C // tile_size, 1), x.dtype, effective bit for each is 16(fp16)
        """
        zeros = zeros.half()
        # zeros_ = zeros.clone()
        target_bit = 8 # fp16 can be divided by 8
        zeros = zeros.view(torch.uint8) # [B, S, C // tile_size, 1] -> [B, S, C // tile_size, 2], uint8
        zeros_bits = cupy.unpackbits(tensor_to_cupy(zeros)).reshape(*zeros.shape, 8)[..., -target_bit:] # (B, S, C // tile_size, 2, 8), bit-stream
        zeros_bits_flatten = zeros_bits.reshape(np.prod(zeros_bits.shape)) # (effective_bit_num, ), uint8
        bits_stream = cupy.concatenate([bits_stream, zeros_bits_flatten], axis=-1)

        """
        scales.
        scales: (B, S, C // tile_size, 1), x.dtype, effective bit for each is 16(fp16)
        """
        scales = scales.half()
        # scales_ = scales.clone()
        target_bit = 8 # fp16 can be divided by 8
        scales = scales.view(torch.uint8) # [B, S, C // tile_size, 1] -> [B, S, C // tile_size, 2], uint8
        scales_bits = cupy.unpackbits(tensor_to_cupy(scales)).reshape(*scales.shape, 8)[..., -target_bit:] # (B, S, C // tile_size, 2, 8), bit-stream
        scales_bits_flatten = scales_bits.reshape(np.prod(scales_bits.shape)) # (effective_bit_num, ), uint8
        bits_stream = cupy.concatenate([bits_stream, scales_bits_flatten], axis=-1)

        """
        token_bit_map.
        token_bit_map: (B, S), uint8, effective bit for each is 1
        """
        target_bit = 1
        token_bit_map_bits = cupy.unpackbits(tensor_to_cupy(token_bit_map)).reshape(*token_bit_map.shape, 8)[..., -target_bit:] # (B, S, 1, 1), bit-stream
        token_bit_map_bits_flatten = token_bit_map_bits.reshape(np.prod(token_bit_map_bits.shape)) # (effective_bit_num, ), uint8
        bits_stream = cupy.concatenate([bits_stream, token_bit_map_bits_flatten], axis=-1)

        """
        tile_bit_map.
        tile_bit_map: (B * S * C // tile_size), uint8, effective bit for each is 1
        """
        target_bit = 1
        tile_bit_map_bits = cupy.unpackbits(tensor_to_cupy(tile_bit_map)).reshape(*tile_bit_map.shape, 8)[..., -target_bit:] # (B * S * C // tile_size, 1, 1), bit-stream
        tile_bit_map_bits_flatten = tile_bit_map_bits.reshape(np.prod(tile_bit_map_bits.shape)) # (effective_bit_num, ), uint8
        bits_stream = cupy.concatenate([bits_stream, tile_bit_map_bits_flatten], axis=-1)

        """
        pivot_indices.
        pivot_indices: (apply_num_tiles, ), int64, effective bit log2(tile_size)
        """
        if pivot_indices is not None:
            assert tile_bit_map.sum() > 0, "tile_bit_map is all zero, so pivot_indices should be None..."
            pivot_indices_uint8 = pivot_indices.view(torch.uint8) # [apply_num_tiles, ] -> [apply_num_tiles * 64 / 8,], uint8
            pivot_indices_uint8_big_endian = pivot_indices_uint8.flip(dims=[-1]) # [apply_num_tiles * 64 / 8,], uint8
            
            target_bit = math.ceil(log2(self.tile_size))
            pivot_indices_bits = cupy.unpackbits(tensor_to_cupy(pivot_indices_uint8_big_endian)).reshape(*pivot_indices.shape, 64)[..., -target_bit:] # (apply_num_tiles, target_bit), bit-stream
            pivot_indices_bits_flatten = pivot_indices_bits.reshape(np.prod(pivot_indices_bits.shape)) # (effective_bit_num, ), uint8
            bits_stream = cupy.concatenate([bits_stream, pivot_indices_bits_flatten], axis=-1)

        packed = cupy.packbits(bits_stream)
        y = cupy_to_tensor(packed)


        return y        
    
    def unpack_data(self, y):
        B, S, C = self.activ_shape
        device = self.device
        dtype = self.dtype
        tile_size = self.tile_size
        unpacked = cupy.unpackbits(tensor_to_cupy(y)) # (n * 8, )

        ptr = 0

        """
        activation.
        --------- low precision part ---------
            bit-width:
                self.low_precision_bits
            element-number:
                B * int(S * (1 - self.high_precision_allocation_ratio)) * C
            total-bit-num:
                low_precision_bits * element_number
        --------- high precision part ---------
            bit-width:
                self.high_precision_bits
            element-number:
                B * int(S * self.high_precision_allocation_ratio) * C
            total-bit-num:
                high_precision_bits * element_number
        """
        low_prec_token_num = int(S * (1 - self.high_precision_allocation_ratio))
        high_prec_token_num = S - low_prec_token_num

        src_bit = self.low_precision_bits
        pack_bit = min(src_bit, 8)
        bit_num = B * low_prec_token_num * C * src_bit
        t_unpacked = unpacked[ptr:ptr + bit_num]
        t_unpacked = t_unpacked.reshape(-1, pack_bit) # (B * low_prec_token_num * C, src_bit)
        y = cupy.pad(t_unpacked, ((0,0), (8 - pack_bit, 0))) # (B * low_prec_token_num * C, 8)
        y = cupy_to_tensor(cupy.packbits(y)) # (B * low_prec_token_num * C, ), uint8
        x_low = y.view(B, low_prec_token_num, C // tile_size, tile_size) # (B, low_prec_token_num, C // tile_size, tile_size), uint8
        ptr += bit_num

        src_bit = self.high_precision_bits
        pack_bit = min(src_bit, 8)
        bit_num = B * high_prec_token_num * C * src_bit
        t_unpacked = unpacked[ptr:ptr + bit_num]
        t_unpacked = t_unpacked.reshape(-1, pack_bit) # (B * high_prec_token_num * C, src_bit)
        y = cupy.pad(t_unpacked, ((0,0), (8 - pack_bit, 0))) # (B * high_prec_token_num * C, 8)
        y = cupy_to_tensor(cupy.packbits(y)) # (B * high_prec_token_num * C, ), uint8
        x_high = y.view(B, high_prec_token_num, C // tile_size, tile_size) # (B, high_prec_token_num, C // tile_size, tile_size), uint8
        ptr += bit_num

        x = torch.cat([x_low, x_high], dim=1) # (B, S, C // tile_size, tile_size), uint8
        # x = x.to(device=device, dtype=dtype)

        """
        zeros.
        zeros: (B, S, C // tile_size, 1), x.dtype, effective bit for each is 16(fp16)
        """
        src_bit = 16
        pack_bit = min(src_bit, 8)
        bit_num = B * S * C // tile_size * src_bit
        t_unpacked = unpacked[ptr:ptr + bit_num]
        t_unpacked = t_unpacked.reshape(-1, pack_bit) # (B * S * C // tile_size * 2, pack_bit)
        y = cupy.pad(t_unpacked, ((0,0), (8 - pack_bit, 0))) # (B * S * C // tile_size * 2, pack_bit)
        # y = y.reshape(B, S, C // tile_size, 2, 8)
         # (B * S * C // tile_size, 2, 8)
        y = cupy_to_tensor(cupy.packbits(y)) # (B * S * C // tile_size * 2, ), uint8
        y = y.reshape(B, S, C // tile_size, 2) # (B, S, C // tile_size, 2, ), uint8
        zeros = y.view(torch.float16) # (B, S, C // tile_size, 1), fp16
        ptr += bit_num

        """
        scales.
        scales: (B, S, C // tile_size, 1), x.dtype, effective bit for each is 16(fp16)  
        """
        src_bit = 16
        pack_bit = min(src_bit, 8)
        bit_num = B * S * C // tile_size * src_bit
        t_unpacked = unpacked[ptr:ptr + bit_num]
        t_unpacked = t_unpacked.reshape(-1, pack_bit) # (B * S * C // tile_size * 2, pack_bit)
        y = cupy.pad(t_unpacked, ((0,0), (8 - pack_bit, 0))) # (B * S * C // tile_size * 2, pack_bit)
        # y = y.reshape(B, S, C // tile_size, 2, 8)
         # (B * S * C // tile_size, 2, 8)
        y = cupy_to_tensor(cupy.packbits(y)) # (B * S * C // tile_size * 2, ), uint8
        y = y.reshape(B, S, C // tile_size, 2) # (B, S, C // tile_size, 2, ), uint8
        scales = y.view(torch.float16) # (B, S, C // tile_size, 1), fp16
        ptr += bit_num

        """
        token_bit_map, (B, S), uint8
        """
        src_bit = 1
        pack_bit = min(src_bit, 8)
        bit_num = B * S * src_bit
        t_unpacked = unpacked[ptr:ptr + bit_num]
        t_unpacked = t_unpacked.reshape(-1, pack_bit) # (B * S, 1)
        y = cupy.pad(t_unpacked, ((0,0), (8 - pack_bit, 0))) # (B * S, 8)
        y = cupy_to_tensor(cupy.packbits(y)) # (B * S, ), uint8
        token_bit_map = y.view(B, S) # (B, S), uint8
        ptr += bit_num

        """
        tile_bit_map, (B * S * C // tile_size, ), uint8
        """
        src_bit = 1
        pack_bit = min(src_bit, 8)
        bit_num = B * S * C // tile_size * src_bit
        t_unpacked = unpacked[ptr:ptr + bit_num]
        t_unpacked = t_unpacked.reshape(-1, pack_bit) # (B * S * C // tile_size, 1)
        y = cupy.pad(t_unpacked, ((0,0), (8 - pack_bit, 0))) # (B * S * C // tile_size, 8)
        y = cupy_to_tensor(cupy.packbits(y)) # (B * S * C // tile_size, ), uint8
        tile_bit_map = y.view(B * S * C // tile_size) # (B * S * C // tile_size, ), uint8
        ptr += bit_num

        """
        pivot_indices, (nonzero_num, ), int64
        """
        src_bit = math.ceil(log2(self.tile_size))
        pack_bit = 64
        apply_num_tiles = int(tile_bit_map.sum())
        pivot_indices = None
        if apply_num_tiles > 0:
            t_unpacked = unpacked[ptr:ptr + apply_num_tiles * src_bit]
            t_unpacked = t_unpacked.reshape(-1, src_bit) # (apply_num_tiles, src_bit)
            y = cupy.pad(t_unpacked, ((0,0), (pack_bit - src_bit, 0))) # (apply_num_tiles, pack_bit)
            y = cupy_to_tensor(cupy.packbits(y)) # (apply_num_tiles * pack_bit / 8), uint8
            y_small_endian = y.flip(dims=[-1]) # (apply_num_tiles * pack_bit / 8), uint8
            
            pivot_indices = y_small_endian.view(torch.int64) # (apply_num_tiles, )
            
            ptr += apply_num_tiles * src_bit

        
        """
        x, (B, S, C // tile_size, tile_size), uint8
        zeros, (B, S, C // tile_size, 1), fp16
        scales, (B, S, C // tile_size, 1), fp16
        token_bit_map, (B, S), uint8
        tile_bit_map, (B * S * C // tile_size, ), uint8
        pivot_indices, (apply_num_tiles, ), int64
        """
        return x, zeros, scales, token_bit_map, tile_bit_map, pivot_indices
        
        

    def compress_send(self, x, i_micro_batch, comm, dst, stream):
        if not flag.FLAG_DISABLE_COMPRESSION:
            with stream:
                _data = self.compress(x,)
                _packed_data = self.pack_data(*_data)
            comm.send(_packed_data, dst=dst, stream=stream)
        else:
            comm.send(x, dst=dst, stream=stream)

    def recv_decompress(self, i_micro_batch, comm, src, stream):
        if not flag.FLAG_DISABLE_COMPRESSION:
            _recv_buffer = self.buffers[i_micro_batch]
            comm.recv(_recv_buffer, src=src, stream=stream)
            with stream:
                _unpacked_x = self.unpack_data(_recv_buffer)
                x = self.decompress(*_unpacked_x,)
            # flush
            _recv_buffer.zero_()
            return x
        else:
            recv_buffer = self.warmup_buffers[i_micro_batch]
            comm.recv(recv_buffer, src=src, stream=stream)
            return recv_buffer