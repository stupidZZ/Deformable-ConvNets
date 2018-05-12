

import cPickle
import os
import time
import mxnet as mx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from module import MutableModule
from utils import image
from bbox.bbox_transform import bbox_pred, clip_boxes
from nms.nms import py_nms_wrapper, py_softnms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from utils.PrefetchingIter import PrefetchingIter
from bbox.bbox_transform import bbox_overlaps
import copy
import math

def im_detect_v2(predictor, data_batch, data_names, scales, cfg):
    output_all = predictor.predict(data_batch)

    data_dict_all = [dict(zip(data_names, idata)) for idata in data_batch.data]
    scores_all = []
    pred_boxes_all = []
    for output, data_dict, scale in zip(output_all, data_dict_all, scales):
        if cfg.TEST.HAS_RPN:
            rois = output['rois_output'].asnumpy()[:, 1:]
        else:
            rois = data_dict['rois'].asnumpy().reshape((-1, 5))[:, 1:]
        im_shape = data_dict['data'].shape

        # save output
        scores = output['cls_prob_reshape_output'].asnumpy()[0]
        bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]

        softmax_1_0 = output['softmax_1_0_output'].asnumpy()
        #softmax_1_1 = output['softmax_1_1_output'].asnumpy()
        softmax_2_0 = output['softmax_2_0_output'].asnumpy()
        #softmax_2_1 = output['softmax_2_1_output'].asnumpy()
        plt.figure(1)

        #plt.show()
        plt.imshow(softmax_1_0)
        #plt.figure(2)
        #plt.imshow(softmax_2_0, cmap='jet')
        #plt.show()
        #plt.imshow(softmax_2_1)

        #key_1_0 = output['key_1_0_output'].asnumpy()
        #key_2_0 = output['key_2_0_output'].asnumpy()
        #query_1_0 = output['query_1_0_output'].asnumpy()
        #query_2_0 = output['query_2_0_output'].asnumpy()

        #key_1_norm = np.linalg.norm(key_1_0, axis=1)
        #key_2_norm = np.linalg.norm(key_2_0, axis=1)
        #query_1_norm = np.linalg.norm(query_1_0, axis=1)
        #query_2_norm = np.linalg.norm(query_2_0, axis=1)
        #plt.figure(1)
        #plt.imshow(softmax_1_0, cmap='jet')

        fc_new_1 = output['fc_new_1_output'].asnumpy()[0]
        mean_val1 = np.mean(fc_new_1)
        std_val1 = np.std(fc_new_1)

        fc_new_2 = output['fc_new_2_output'].asnumpy()[0]
        mean_val2 = np.mean(fc_new_2)
        std_val2 = np.std(fc_new_2)

        attention_1 = output['reshape7_output'].asnumpy()[0]
        mean_val3 = np.mean(attention_1)
        std_val3 = np.std(attention_1)

        attention_2 = output['reshape15_output'].asnumpy()[0]
        mean_val4 = np.mean(attention_2)
        std_val4 = np.std(attention_2)

        # post processing
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_shape[-2:])

        # we used scaled image & roi to train, so it is necessary to transform them back
        pred_boxes = pred_boxes / scale

        scores_cls_id = np.argmax(scores, axis=1)
        scores_max = scores.max(axis=1)

        summed_softmax = np.sum(softmax_1_0, axis=0)

        sorted_ids = np.argsort(summed_softmax)[::-1]

        for id in range(20):
            print str(id), '=', str(sorted_ids[id]), ', sum=', str(summed_softmax[sorted_ids[id]]), ', rois=', \
                rois[sorted_ids[id]], ', cls_id=', scores_cls_id[sorted_ids[id]], ',cls_score=', scores_max[sorted_ids[id]]

        scores_all.append(scores)
        pred_boxes_all.append(pred_boxes)
    return scores_all, pred_boxes_all, data_dict_all

def py_gtnms_wrapper(thresh):
    def _nms(dets, gts):
        return gt_nms(dets, gts, thresh)
    return _nms

def gt_nms(dets, gts, thresh):
    if dets.shape[0] == 0:
        return np.zeros((0,5))

    ious_thres_min = 0.2

    softnms = py_softnms_wrapper(ious_thres_min)
    softnms_02 = py_softnms_wrapper(0.2)
    if gts.shape[0] == 0:
        return softnms_02(dets)

    #maxDets = min(100, dets.shape[0])
    #dets = dets[0:maxDets, :]
    scores = dets[:,4]
    sids = scores.argsort()[::-1]
    dets = dets[sids,:]
    overlap = bbox_overlaps(dets[:, 0:4].astype(np.float), gts[:, 0:4].astype(np.float))


    ious_thres = 0.5
    gt_max = - np.ones((gts.shape[0]), dtype=np.float)
    gt_iou_first = - np.ones((gts.shape[0]), dtype=np.float)
    gtm = - np.ones((gts.shape[0]), dtype=np.float)

    gtm2 = - np.ones((gts.shape[0]), dtype=np.float)
    dtm = - np.ones((dets.shape[0]), dtype=np.float)
    for dind, d in enumerate(dets):
        # information about best match so far (m=-1 -> unmatched)
        iou = min([ious_thres, 1 - 1e-10])
        iou_score = 0.001
        m = -1
        for gind, g in enumerate(gts):
            # if this gt already matched, and not a crowd, continue
            if gt_max[gind] > 0.9499:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1:
                break
            # continue to next gt unless better match made
            if overlap[dind, gind] < iou or overlap[dind, gind] < gt_max[gind]+0.01: # or overlap[dind, gind] * dets[dind, 4] < iou_score:
                continue
            # match successful and best so far, store appropriately
            iou = overlap[dind, gind]
            #iou_score = overlap[dind, gind] * dets[dind, 4]
            m = gind
        # if match made store id of match for both dt and gt
        if m == -1:
            continue
        if gtm[m] < -0.99:
            gtm[m] = dind
            dtm[dind] = m

        if gt_iou_first[m] < -0.99:
            gt_iou_first[m] = overlap[dind, m]
        gt_max[m] = overlap[dind, m]
        if overlap[dind, gind] > np.floor(gt_iou_first[m] *20 + 3) / 20.0:
            gtm2[m] = dind

    for gind, g in enumerate(gts):
        if gtm2[gind] >= 0:
            dtm[gtm2[gind]] = gind



    dt_select = np.where(dtm > -0.99)[0]
    #dets_ori = copy.deepcopy(dets)
    #dets_new = softnms(dets)
    #dets_new[dt_select, 4] = dets_ori[dt_select, 4]

    dt_nonselect = np.where(dtm <= -0.99)[0]

    if len(dt_select) > 0 and len(dt_nonselect) > 0:
        overlap_dt = bbox_overlaps(dets[dt_select, 0:4].astype(np.float), dets[dt_nonselect, 0:4].astype(np.float))

        dets_nonselect = dets[dt_nonselect, :]
        dets_nonselect = softnms(dets_nonselect)
        max_overlap = np.amax(overlap_dt, axis=0)
        dets_nonselect[:, 4] = dets_nonselect[:, 4] * np.exp(-max_overlap ** 2 / ious_thres_min)

        dets_new = np.concatenate((dets[dt_select, :], dets_nonselect), axis=0)

        #dets_left = dets[dt_nonselect[np.where(max_overlap < ious_thres_min)[0]], :]
        #dets = np.concatenate((dets[dt_select, :], dets_left), axis=0)
        #dets_ori = copy.deepcopy(dets)

        #dets_new[0:len(dt_select), 4] = dets_ori[0:len(dt_select), 4]
    else:
        dets_ori = copy.deepcopy(dets)
        dets_new = softnms_02(dets)
        dets_new[dt_select, 4] = dets_ori[dt_select, 4]
    #scores = dets[:,4]
    #sort_ids = np.argsort(scores)[::-1]
    #dets_new = dets[sort_ids, :]

    return dets_new
