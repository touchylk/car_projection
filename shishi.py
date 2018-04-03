# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import tensorflow as tf
import datetime
import os
import argparse
import yolo.config as cfg
import matplotlib.pyplot as plt
from yolo.yolo_net import YOLONet
from utils.timer import Timer
from utils.pascal_voc_verclass import pascal_voc
import xml.etree.ElementTree as ET
import cv2
import Image

dir = '/home/e813/image/'
num = 0
for i in os.listdir(dir):
    num+=1
    imname = os.path.join(dir,i)
    im = cv2.imread(imname)
    nam = os.path.join(dir,'out',str(num)+'.jpg')
    cv2.imwrite(nam,im)


"""def pre_process(dir_file,num):
    x_left = 2
    x_right = 20
    y_up = 30
    y_down = 5
    output_dir = '/media/e813/E/pre_proceed_7/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tree = ET.parse(dir_file)
    imname = tree.find('path').text
    im = cv2.imread(imname)
    im2 = im[y_up:-y_down, x_left:-x_right, :]
    root = tree.getroot()
    save = True
    for obj in root.iter('object'):
        for bnd in obj.iter('bndbox'):
            for xmin in bnd.iter('xmin'):
                xmin_2 = (int(xmin.text)-x_left)
                if xmin_2<1:
                    save = False
                    continue
                    xmin_2=1
                xmin.text = str(xmin_2)
            for xmax in bnd.iter('xmax'):
                xmax_2 = (int(xmax.text)-x_left)
                if xmax_2<1:
                    save = False
                    continue
                    xmax_2=1
                if xmax_2>=im2.shape[1]:
                    save = False
                    continue
                    xmax_2 = im2.shape[1]
                xmax.text = str(xmax_2)
            for ymin in bnd.iter('ymin'):
                ymin_2 = (int(ymin.text)-y_up)
                if ymin_2<1:
                    save = False
                    continue
                    ymin_2=1
                ymin.text = str(ymin_2)
            for ymax in bnd.iter('ymax'):
                ymax_2 = (int(ymax.text)-y_up)
                if ymax_2<1:
                    save = False
                    continue
                    ymax_2 = 1
                if ymax_2>=im2.shape[0]:
                    save = False
                    continue
                    ymax_2 = im2.shape[0]-1
                ymax.text = str(ymax_2)
    file_con_name = output_dir+'pre_processed_{}'.format(num)
    if not save:
        return
    imed = file_con_name+'.jpg'
    xmled = file_con_name+'.xml'
    cv2.imwrite(imed, im2)
    tree.write(xmled)



devkil_path = '/media/e813/E/CarPic/'
name1s = os.listdir(devkil_path)
num=0
for name1 in name1s:
        dir1 = os.path.join(devkil_path, name1)
        name2s = os.listdir(dir1)
        for name2 in name2s:
            dir2 = os.path.join(dir1, name2)
            name3s = os.listdir(dir2)
            for name3 in name3s:
                if name3[-3:] == 'xml':
                    num +=1
                    dir_file = os.path.join(dir2,name3)
                    pre_process(dir_file,num)
"""


