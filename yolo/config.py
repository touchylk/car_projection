# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import datetime
#
# path and dataset parameter
#
LAST_STEP = 6100
AX_LOW = 1.0
AX_HIGHT = 1.6

GPU = '1,0'

PASCAL_PATH = '/media/e813/E/CarPic/'

OUTPUT_dir = 'data/output'

LAYER_to_restore = 48
WEIGHTS_init_dir_file = '/home/e813/yolo_weights/YOLO_small.ckpt'
WEIGHTS_output_dir = '/media/e813/D/weight_output/car_project/proceed_netchang_imagenorm'#2_non_flipped'
TRAIN_process_save_txt_dir = WEIGHTS_output_dir


WEIGHTS_to_save_dir = WEIGHTS_output_dir


WEIGHTS_file_name = 'save.ckpt-{}'.format(LAST_STEP)
CACHE_PATH = 'data/label_cache'
WERIGHTS_READ = os.path.join(WEIGHTS_output_dir,'save.ckpt-{}'.format(LAST_STEP))
WEIGHTS_to_restore = WERIGHTS_READ
#'/media/e813/D/weight_output/car_project/shiyan_1/save.ckpt-5000' #os.path.join(WEIGHTS_output_dir,WEIGHTS_file_name)
#WERIGHTS_READ = '/home/e813/yolo_weights/output/yolo_verclass/save.ckpt-6000'

IMAGE_dir_file =   '/media/e813/E/CarPic/labeled/2017012003002959778588975/2017012003002959778588975_0006068624.jpg'
    #'/media/e813/E/CarPic/labeled/2017012002595416120329698/2017012002595416120329698_0014998415.jpg'
    #'/media/e813/E/dataset/obvious/tri02299/2017021609323901838992190/2017021609323901838992190_0019462671.jpg'#'/home/e813/dataset/VOCdevkit _2007_trainval/VOC2007/JPEGImages/000138.jpg'
IMAGE_output_dir = os.path.join(WEIGHTS_output_dir,'imageout')
if 0:
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
elif 0:
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
elif 1:
    CLASSES = ['headlight', 'tailight', 'tyre']
else:
    CLASSES = ['person']

CLASS_NUM = len(CLASSES)
if CLASS_NUM == 1:
    ONE_CLASS = True
else:
    ONE_CLASS = False

FLIPPED = True


#
# model parameter
#

IMAGE_SIZE = 448

CELL_SIZE = 7

BOXES_PER_CELL = 2

ALPHA = 0.1

DISP_CONSOLE = False

OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 0.1
CLASS_SCALE = 1.0
COORD_SCALE = 5.0


#
# solver parameter
#



LEARNING_RATE = 0.0001

DECAY_STEPS = 800

DECAY_RATE = 0.316

STAIRCASE = True

BATCH_SIZE = 20

MAX_ITER = 3000

SUMMARY_ITER = 2

SAVE_ITER = 300


#
# test parameter
#

THRESHOLD = 0.1


IOU_THRESHOLD = 0.2

"""
1,文件记录每一次训练用到了第几张图片,每一个epoch,将shuffle后的标签储存,使得每一次重新训练都能像继续训练一样
2,学习结构化保存对象的包,储存训练的每一个过程.
3,双GPU分配batch
"""