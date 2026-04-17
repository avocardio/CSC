from .quantization import (
    CompressionGranularity,
    DifferentiableQuantizer,
    quantize,
    compute_average_bit_depth,
    get_quantizers,
    get_compression_stats,
)
from .resnet import QuantizedResNet18, QuantizedConv2d, BasicBlock
from .compression import (
    apply_bias_l1_penalty,
    find_removable_channels,
    remove_dead_channels,
)
