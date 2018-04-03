# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
import yolo.config as cfg
from yolo.yolo_net import YOLONet
from utils.timer import Timer

def interpret_output(self, output):
    """

    :param output:
    :return:
    predict_scales = tf.reshape(predicts[:, :self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
        # predict_scales为(batch_size,7,7,2)
    predict_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

    """

    probs = np.zeros((self.cell_size, self.cell_size,
                      self.boxes_per_cell))
    iclass_probs = np.zeros((self.cell_size, self.cell_size,
                             self.boxes_per_cell, self.num_class))
    iscales = np.zeros((self.cell_size, self.cell_size,
                        self.boxes_per_cell, self.num_class))
    # class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))
    # 将net输出的IOU,reshape成(7,7,2)维
    scales = np.reshape(output[0:self.boundary2], (self.cell_size, self.cell_size, self.boxes_per_cell))

    # 将net输出的坐标,reshape成(7,7,2,4)维,boxes
    boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
    offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                     [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))

    boxes[:, :, :, 0] += offset
    boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
    boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, 0:2] / self.cell_size
    boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
    # 到这里,boxes为(7,7,2,4),(:,:,:,0:1)为0-1之间,表示整张图的相对位置
    # (:,:,:,2：3)

    boxes *= self.image_size
    # 到这里,boxes为(7,7,2,4),整张图的绝对位置,0-448

    for i in range(self.boxes_per_cell):
        probs[:, :, i] = scales[:, :, i]
        iscales[:, :, i] = scales[:, :, i]
        # probs为(7,7,2,1)维,值为IOU

    filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)

    # print filter_mat_boxes[0]
    boxes_filtered = boxes[filter_mat_boxes[0],
                           filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    # iclass_probs_filtered = iclass_probs[filter_mat_probs]
    # iscales_filtered = iscales[filter_mat_probs]

    classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
                                                                   0], filter_mat_boxes[1], filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]
    # iclass_probs_filtered = iclass_probs_filtered[argsort]
    # iscales_filtered = iscales_filtered[argsort]


    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                probs_filtered[j] = 0.0
                # iclass_probs_filtered[j] = 0.0
                # iscales_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    print filter_iou
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]
    # iclass_probs_filtered = iclass_probs_filtered[filter_iou]
    # iscales_filtered = iscales_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append([self.classes[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
            i][1], boxes_filtered[i][2], boxes_filtered[i][3],
                       probs_filtered[i]])  # need to define iclass_probs_filtered and iscales_filtered

    return result