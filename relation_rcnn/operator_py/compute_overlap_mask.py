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

DEBUG = False


class ComputeOverlapMaskOperator(mx.operator.CustomOp):
    def __init__(self):
        super(ComputeOverlapMaskOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        rois_left = in_data[0].asnumpy()
        rois_right = in_data[1].asnumpy()
        rois_len_left = rois_left.shape[0]
        rois_len_right = rois_right.shape[0]
        overlap = bbox_overlaps(rois_left[:,1:].astype(np.float), rois_right[:,1:].astype(np.float))
        mask = (overlap > 0.01).astype(np.float)

        for ind, val in enumerate([mask, overlap]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('compute_overlap_mask')
class ComputeOverlapMaskProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ComputeOverlapMaskProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['rois_left', 'rois_right']

    def list_outputs(self):
        return ['mask', 'overlap']

    def infer_shape(self, in_shape):
        rois_left_shape = in_shape[0]
        rois_right_shape = in_shape[1]
        mask_shape = (rois_left_shape[0], rois_right_shape[0])
        overlap_shape = (rois_left_shape[0], rois_right_shape[0])

        return [rois_left_shape, rois_right_shape], \
               [mask_shape, overlap_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ComputeOverlapMaskOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
