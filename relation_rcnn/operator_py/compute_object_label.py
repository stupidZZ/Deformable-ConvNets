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



class ComputeObjectLabelOperator(mx.operator.CustomOp):
    def __init__(self):
        super(ComputeObjectLabelOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        label = in_data[0].asnumpy()
        object_label = (label > 0).astype(np.float)

        for ind, val in enumerate([object_label]):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register('compute_object_label')
class ComputeObjectLabelProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ComputeObjectLabelProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['label']

    def list_outputs(self):
        return ['object_label']

    def infer_shape(self, in_shape):
        label_shape = in_shape[0]
        object_label_shape = label_shape

        return [label_shape], \
               [object_label_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ComputeObjectLabelOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
