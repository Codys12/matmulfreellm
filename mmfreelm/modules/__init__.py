# -*- coding: utf-8 -*-

from mmfreelm.modules.convolution import (ImplicitLongConvolution, LongConvolution,
                                          ShortConvolution)
from mmfreelm.modules.fused_cross_entropy import FusedCrossEntropyLoss
from mmfreelm.modules.fused_mse import FusedSoftCrossEntropyLoss
from mmfreelm.modules.fused_norm_gate import FusedRMSNormSwishGate
from mmfreelm.modules.layernorm import (LayerNorm, LayerNormLinear, RMSNorm,
                                        RMSNormLinear)

__all__ = [
    'ImplicitLongConvolution', 'LongConvolution', 'ShortConvolution',
    'FusedCrossEntropyLoss', 'FusedSoftCrossEntropyLoss',
    'LayerNorm', 'LayerNormLinear', 'RMSNorm', 'RMSNormLinear',
    'FusedRMSNormSwishGate'
]
