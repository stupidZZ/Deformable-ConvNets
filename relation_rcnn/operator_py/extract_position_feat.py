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
import cPickle

DEBUG = False


class ExtractPositionFeatOperator(mx.operator.CustomOp):
    def __init__(self, feat_dim=1024):
        super(ExtractPositionFeatOperator, self).__init__()
        self._feat_dim = feat_dim
        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0].asnumpy()

        rois_center = (rois[:,1:3]+rois[:,3:5]) * 0.5
        rois_wh = rois[:,3:5] - rois[:,1:3] + 1
        dim_mat = np.tile(np.arange(self._feat_dim/8),[rois.shape[0], 1])
        dim_mat = 1000.0 ** (8.0/self._feat_dim * dim_mat)
        x_mat = np.transpose(np.tile(rois_center[:,0]/4,[self._feat_dim/8, 1]))
        y_mat = np.transpose(np.tile(rois_center[:,1]/4,[self._feat_dim/8, 1]))
        w_mat = np.transpose(np.tile(rois_wh[:,0],[self._feat_dim/8, 1]))
        h_mat = np.transpose(np.tile(rois_wh[:,1],[self._feat_dim/8, 1]))
        position_feat = np.hstack((np.sin(x_mat / dim_mat), np.cos(x_mat/dim_mat), np.sin(y_mat/dim_mat), np.cos(y_mat/dim_mat), 
                np.sin(w_mat / dim_mat), np.cos(w_mat/dim_mat), np.sin(h_mat/dim_mat), np.cos(h_mat/dim_mat)))
 
        if DEBUG:
            print position_feat
        for ind, val in enumerate([position_feat]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register('extract_position_feat')
class ExtractPositionFeatProp(mx.operator.CustomOpProp):
    def __init__(self, feat_dim=1024):
        super(ExtractPositionFeatProp, self).__init__(need_top_grad=False)
        self._feat_dim = int(feat_dim)

    def list_arguments(self):
        return ['rois']

    def list_outputs(self):
        return ['position_feat']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        rois = rpn_rois_shape[0]

        position_feat_shape = (rois, self._feat_dim)

        return [rpn_rois_shape], \
               [position_feat_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ExtractPositionFeatOperator(self._feat_dim)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
