from .dummy_modules import NoCompression
from .fixpoint_modules import FixpointCompressor, FixpointFlexibleCompressor
from .delta_modules import DeltaCompressor, DeltaLowBitsCompressor
from .tah_modules import TAHQuantCompressor, NaiveCompressor

def get_compressor(*args, compress_method='none', **kargs):
    if compress_method == 'none':
        return NoCompression(*args, **kargs)
    elif compress_method == 'fixpoint':
        return FixpointFlexibleCompressor(*args, **kargs)
    elif compress_method == 'delta':
        return DeltaCompressor(*args, **kargs)
    elif compress_method == 'delta-lowbits':
        return DeltaLowBitsCompressor(*args, **kargs)
    elif compress_method == 'tah':
        return TAHQuantCompressor(*args, **kargs)
    elif compress_method == 'naive':
        return NaiveCompressor(*args, **kargs)
    else:
        raise Exception('unknown compression method')