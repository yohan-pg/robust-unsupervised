# Standard libraries
import numpy as np

# PyTorch
import torch
import torch.nn as nn


# static const unsigned int std_luminance_quant_tbl[DCTSIZE2] = {
#   16,  11,  10,  16,  24,  40,  51,  61,
#   12,  12,  14,  19,  26,  58,  60,  55,
#   14,  13,  16,  24,  40,  57,  69,  56,
#   14,  17,  22,  29,  51,  87,  80,  62,
#   18,  22,  37,  56,  68, 109, 103,  77,
#   24,  35,  55,  64,  81, 104, 113,  92,
#   49,  64,  78,  87, 103, 121, 120, 101,
#   72,  92,  95,  98, 112, 100, 103,  99
# };

y_table = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float32,
).T


y_table = nn.Parameter(torch.from_numpy(y_table))
#
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array(
    [[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]]
).T
c_table = nn.Parameter(torch.from_numpy(c_table))


# static const unsigned int std_chrominance_quant_tbl[DCTSIZE2] = {
#   16,  18,  24,  47,  99,  99,  99,  99,
#   18,  21,  26,  66,  99,  99,  99,  99,
#   24,  26,  56,  99,  99,  99,  99,  99,
#   47,  66,  99,  99,  99,  99,  99,  99,
#   99,  99,  99,  99,  99,  99,  99,  99,
#   99,  99,  99,  99,  99,  99,  99,  99,
#   99,  99,  99,  99,  99,  99,  99,  99,
#   99,  99,  99,  99,  99,  99,  99,  99
# };


class SurrogateDiffRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        ctx.save_for_backward(x, k)
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_y):
        (x, k) = ctx.saved_tensors
        with torch.enable_grad():
            approx = torch.round(x) + (x - torch.round(x))**3
            return (
                torch.autograd.grad(
                    x.lerp(approx, k.to(x.device)), x, grad_y 
                )[0], 
                None,
            )


def diff_round(k):
    """Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return lambda x: SurrogateDiffRound.apply(x, torch.tensor(k))


def quality_to_factor(quality):
    """Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000.0 / quality
    else:
        quality = 200.0 - quality * 2
    return quality / 100.0
