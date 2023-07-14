# Pytorch
import torch
import torch.nn as nn

# Local
from modules import compress_jpeg, decompress_jpeg
import utils
from utils import diff_round, quality_to_factor
import threading


class DiffJPEG(nn.Module):
    def __init__(self, k, quantization_table, differentiable=True):
        """Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme.
        """
        super().__init__()
        if differentiable:
            rounding_y = diff_round(k)
            rounding_c = diff_round(k)
        else:
            rounding_y = torch.round
            rounding_c = torch.round
        self.compress = compress_jpeg(
            rounding_y=rounding_y,
            rounding_c=rounding_c,
            quantization_table=quantization_table,
        )
        self.decompress = decompress_jpeg(quantization_table=quantization_table)

    def parameters(self, recurse=False):
        return []

    def forward(self, x):
        """ """
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr)
        return recovered
