# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSDeformAttn_split(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64
        
        self.d_model = d_model
        self.half_d_model = d_model // 2
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)

        self.value_proj1 = nn.Linear(self.half_d_model, self.half_d_model)
        self.value_proj2 = nn.Linear(self.half_d_model, self.half_d_model)
        
        self.output_proj1 = nn.Linear(self.half_d_model, self.half_d_model)
        self.output_proj2 = nn.Linear(self.half_d_model, self.half_d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)

        xavier_uniform_(self.value_proj1.weight.data)
        constant_(self.value_proj1.bias.data, 0.)
        xavier_uniform_(self.value_proj2.weight.data)
        constant_(self.value_proj2.bias.data, 0.)

        xavier_uniform_(self.output_proj1.weight.data)
        constant_(self.output_proj1.bias.data, 0.)
        xavier_uniform_(self.output_proj2.weight.data)
        constant_(self.output_proj2.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape # (N, num_query, C)
        N, Len_in, _ = input_flatten.shape # N, ∑L(HxW), C，（所有的特征图分辨率拉平的加和）
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value_1 = input_flatten[:, :, :self.half_d_model]
        value_2 = input_flatten[:, :, self.half_d_model:]
        value_1 = self.value_proj1(value_1).view(N, Len_in, self.n_heads, self.half_d_model // self.n_heads)
        value_2 = self.value_proj1(value_2).view(N, Len_in, self.n_heads, self.half_d_model // self.n_heads)
        value = torch.cat([value_1, value_2], dim=-1).view(N, Len_in, self.d_model)

        if input_padding_mask is not None: # input_padding_mask为 (N, ∑L (HxW))，如果是padding，则该像素位置为True
            value = value.masked_fill(input_padding_mask[..., None], float(0)) # 这里的操作是将value中的padding部分置为0
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads) # （N, ∑L(HxW), head, model//head）
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2) # Sampling_offsets为(N, num_query, head, 尺度, num_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points) # 在每个head中，不同尺度level中共包含的参考点的注意力加和为1。这里的attention_weights为(N, num_query, head, 尺度, num_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        # ⬇️ 对sampling_locations进行offset偏移修正
        if reference_points.shape[-1] == 2: # 如果参考点最后是2D的
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1) # 转化为W，H concat 在一起
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4: # 如果参考点最后是4D的
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        output = MSDeformAttnFunction.apply( # 输入  拉平的不同level特征图的线性映射后的value值；不同尺度下的特征图长宽；在拉平的input中不同尺度特征图的起始位置；偏移后的sampling点；query线性映射后的注意力权重；64
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        

        # 重组，例如(1,2),(1,2),(1,2),(1,2) -> (1111),(2222)
        output = output.view(N, Len_q, self.n_heads, self.d_model // self.n_heads)

        output_1 = output[:, :, :, :self.half_d_model//self.n_heads].contiguous().view(N, Len_q, self.half_d_model)
        output_2 = output[:, :, :, self.half_d_model//self.n_heads:].contiguous().view(N, Len_q, self.half_d_model)
        output_1 = self.output_proj1(output_1)
        output_2 = self.output_proj2(output_2)
        output = torch.cat([output_1, output_2], dim=-1) # (N, Len_q, self.d_model)

        return output
