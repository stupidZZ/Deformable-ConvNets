"""
ProposalIm Operator transform anchor coordinates into ROI coordinates with prediction results on
classification probability and bounding box prediction results, and image size and scale information.
"""

import mxnet as mx
import numpy as np
import numpy.random as npr

class ProposalImOperator(mx.operator.CustomOp):
    def __init__(self):
        super(ProposalImOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):

        im_info = in_data[0].asnumpy()[0, :]

        roi_im = np.zeros((1, 5), dtype=np.float32)
        roi_im[0, 3] = im_info[1]-1
        roi_im[0, 4] = im_info[0]-1
        #print im_info
        self.assign(out_data[0], req[0], roi_im)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)

@mx.operator.register("proposal_im")
class ProposalImProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(ProposalImProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['im_info']

    def list_outputs(self):
        return ['roi_im']

    def infer_shape(self, in_shape):
        im_info_shape = in_shape[0]

        roi_im_shape = (1, 5)
        return [im_info_shape], [roi_im_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return ProposalImOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
