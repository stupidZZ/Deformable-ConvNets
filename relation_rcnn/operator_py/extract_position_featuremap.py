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


class ExtractPositionFeaturemapOperator(mx.operator.CustomOp):
    def __init__(self, feat_dim=1024):
        super(ExtractPositionFeaturemapOperator, self).__init__()
        self._feat_dim = feat_dim

    def forward(self, is_train, req, in_data, out_data, aux):
        feat_map = in_data[0].asnumpy()
        height = feat_map.shape[2]
        width = feat_map.shape[3]
        xs = np.transpose(np.tile(np.arange(width), [self._feat_dim/4, 1]))
        ys = np.transpose(np.tile(np.arange(height), [self._feat_dim/4, 1]))

        dim_mat_xs = np.tile(np.arange(self._feat_dim/4),[width, 1])
        dim_mat_ys = np.tile(np.arange(self._feat_dim/4),[height, 1])
        dim_mat_xs = 10000.0 ** (4.0/self._feat_dim * dim_mat_xs)
        dim_mat_ys = 10000.0 ** (4.0/self._feat_dim * dim_mat_ys)

        position_feat_x = np.hstack((np.sin(xs / dim_mat_xs), np.cos(xs / dim_mat_xs)))
        position_feat_x_repeat = np.tile(np.reshape(position_feat_x, (1,self._feat_dim/2, 1, width)),[1,1, height, 1])
        position_feat_y = np.hstack((np.sin(ys / dim_mat_ys), np.cos(ys / dim_mat_ys)))
        position_feat_y_repeat = np.tile(np.reshape(position_feat_y, (1,self._feat_dim/2, height, 1)), [1, 1, 1, width])
        position_feat = np.concatenate((position_feat_x_repeat, position_feat_y_repeat), axis=1) 
 
        if DEBUG:
            print position_feat
        for ind, val in enumerate([position_feat]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register('extract_position_featuremap')
class ExtractPositionFeaturemapProp(mx.operator.CustomOpProp):
    def __init__(self, feat_dim=1024):
        super(ExtractPositionFeaturemapProp, self).__init__(need_top_grad=False)
        self._feat_dim = int(feat_dim)

    def list_arguments(self):
        return ['rois']

    def list_outputs(self):
        return ['position_feat']

    def infer_shape(self, in_shape):
        feature_shape = in_shape[0]

        position_feat_shape = (1, self._feat_dim, feature_shape[2], feature_shape[3])

        return [feature_shape], \
               [position_feat_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ExtractPositionFeaturemapOperator(self._feat_dim)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
