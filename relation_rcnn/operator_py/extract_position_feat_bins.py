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


class ExtractPositionFeatBinsOperator(mx.operator.CustomOp):
    def __init__(self, feat_dim=256, h_nbins=7, w_nbins=7):
        super(ExtractPositionFeatBinsOperator, self).__init__()
        self._h_nbins=h_nbins
        self._w_nbins=w_nbins
        self._feat_dim = feat_dim

    def forward(self, is_train, req, in_data, out_data, aux):
        rois = in_data[0].asnumpy()
        w_nbins = int(self._w_nbins)
        h_nbins = int(self._h_nbins)
        feat_dim = int(self._feat_dim)
        rois_x = np.transpose(np.tile(np.arange(w_nbins),[feat_dim/4, h_nbins]))
        rois_y = np.transpose(np.tile(np.transpose(np.tile(np.arange(h_nbins),[w_nbins, 1])).reshape((h_nbins * w_nbins)), [feat_dim/4, 1]))

        dim_mat = np.tile(np.arange(feat_dim/4),[h_nbins * w_nbins, 1])
        dim_mat = 1000.0 ** (4.0/feat_dim * dim_mat)
        position_feat = np.hstack((np.sin(rois_x / dim_mat), np.cos(rois_x/dim_mat), np.sin(rois_y/dim_mat), np.cos(rois_y/dim_mat))).reshape((1, h_nbins * w_nbins, feat_dim))
 
        if DEBUG:
            print position_feat
        for ind, val in enumerate([position_feat]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register('extract_position_feat_bins')
class ExtractPositionFeatBinsProp(mx.operator.CustomOpProp):
    def __init__(self, feat_dim=256, h_nbins=7, w_nbins=7):
        super(ExtractPositionFeatBinsProp, self).__init__(need_top_grad=False)
        self._h_nbins = h_nbins
        self._w_nbins = w_nbins
        self._feat_dim = feat_dim

    def list_arguments(self):
        return ['rois']

    def list_outputs(self):
        return ['position_feat']

    def infer_shape(self, in_shape):
        rois_shape = in_shape[0]

        #position_feat_shape = (1, self._h_nbins*self._w_nbins, self._feat_dim)
        position_feat_shape = (1, int(self._h_nbins)*int(self._w_nbins), int(self._feat_dim))

        return [rois_shape], \
               [position_feat_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ExtractPositionFeatBinsOperator(self._feat_dim)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
