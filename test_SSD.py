import os
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import warnings

warnings.filterwarnings("ignore")
from notebooks import visualization
slim = tf.contrib.slim

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
    
ckpt_filename = r'D:\Vishnu\test_dataset\SSD\log_2\model.ckpt-1471'

print("Model loaded successfully!")

isess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

# Uncomment the line below if you want to restore the model from the checkpoint
# saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

def process_image(img,confidence_threshold=0.4, nms_threshold=0.0, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})
  

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
            rpredictions, rlocalisations, ssd_anchors,
            select_threshold=confidence_threshold, img_shape=net_shape, num_classes=1, decode=True)
    
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)

    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    
    # Filter detections based on confidence threshold
    mask = rscores > confidence_threshold
    rclasses = rclasses[mask]
    rscores = rscores[mask]
    rbboxes = rbboxes[mask]
    
    return rclasses, rscores, rbboxes

path = 'D:/Vishnu/test_dataset/SSD/Dataset/'
image_names = sorted(os.listdir(path))

# Load the last image in the directory
img = cv2.imread(os.path.join(path, image_names[-4]))

rclasses, rscores, rbboxes = process_image(img)

# Visualize the results
visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
plt.show()