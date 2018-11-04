import numpy as np
import torch
import torch.nn as nn


def hard_threshold(arr, thresh=0.0):
    return arr[arr <= thresh] = 0.0


class JumpReLU(nn.Module):
    r"""Applies the shifted rectified linear unit function element-wise
    
    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input):
        if(self.training == True):
            return max(input, 0.0)

        elif(self.training == False):
            return hard_threshold(input, thresh=1.5)

    def __repr__(self):
        return self.__class__.__name__ + '()'
