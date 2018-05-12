# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Han Hu
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np
from distutils.util import strtobool
from easydict import EasyDict as edict
from bbox.bbox_transform import bbox_overlaps
import cPickle


class ComputePositionFeatOperator(mx.operator.CustomOp):
    def __init__(self):
        super(ComputePositionFeatOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        rois_left = in_data[0].asnumpy()
        rois_right = in_data[1].asnumpy()
        rois_len_left = rois_left.shape[0]
        rois_len_right = rois_right.shape[0]
        center_left =(rois_left[:,1:3] + rois_left[:,3:5]) * 0.5
        center_right =(rois_right[:,1:3] + rois_right[:,3:5]) * 0.5
        wh_left = rois_left[:,3:5] - rois_left[:,1:3] + 1
        wh_right = rois_right[:,3:5] - rois_right[:,1:3] + 1

        center_left_reshape = center_left.reshape(rois_len_left, 1, 2)
        center_right_reshape = center_right.reshape(1, rois_len_right, 2)
        center_diff = center_right_reshape - center_left_reshape
        wh_left_reshape = wh_left.reshape(rois_len_left, 1, 2)
        feat_v1 = center_diff / wh_left_reshape
        rois_left_reshape = rois_left[:,1:].reshape(rois_len_left, 1, 4)
        rois_right_reshape = rois_right[:,1:].reshape(1, rois_len_right, 4)
        rois_diff = rois_right_reshape - rois_left_reshape
        feat_v2 = rois_diff

        for ind, val in enumerate([feat_v1, feat_v2]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('compute_position_feat')
class ComputePositionFeatProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ComputePositionFeatProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['rois_left', 'rois_right']

    def list_outputs(self):
        return ['feat_v1', 'feat_v2']

    def infer_shape(self, in_shape):
        rois_left_shape = in_shape[0]
        rois_right_shape = in_shape[1]
        feat_v1_shape = (rois_left_shape[0], rois_right_shape[0], 2)
        feat_v2_shape = (rois_left_shape[0], rois_right_shape[0], 4)

        return [rois_left_shape, rois_right_shape], \
               [feat_v1_shape, feat_v2_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ComputePositionFeatOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
