# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2018 Data Science Bowl
# Find the nuclei in divergent images to advance medical discovery
# https://www.kaggle.com/c/data-science-bowl-2018

# This code is based on the Python scripts of the following web sites.
#
# 1. Keras U-Net starter - LB 0.277
# https://www.kaggle.com/code/keegil/keras-u-net-starter-lb-0-277/notebook
# 2. U-Net Image Segmentation in Keras
# https://androidkt.com/tensorflow-keras-unet-for-image-image-segmentation/


import os
import sys

import numpy as np

from tqdm import tqdm

# pip install scikit-image
from skimage.transform import resize
#from skimage.morphology import label
from skimage.io import imread, imshow

import traceback

class NucleiDataset:

  def __init__(self, resized_image):
    self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS = resized_image

 
  def create(self, data_path="./stage1_train", has_mask=True):

    image_ids = next(os.walk(data_path))[1]
    X = None
    Y = None

    X = np.zeros((len(image_ids), self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
    if has_mask:
      Y = np.zeros((len(image_ids), self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=np.bool)

    for n, id_ in tqdm(enumerate(image_ids), total=len(image_ids)):
      path = data_path + id_
      img = imread(path + '/images/' + id_ + '.png')[:,:,:self.IMG_CHANNELS]
      img = resize(img, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant', preserve_range=True)
      X[n] = img
      if has_mask:
        mask = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
          mask_ = imread(path + '/masks/' + mask_file)
          mask_ = np.expand_dims(resize(mask_, (self.IMG_HEIGHT, self.IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
          mask = np.maximum(mask, mask_)
        Y[n] = mask

    return X, Y


if __name__ == "__main__":
  try:
    resized_image = (128, 128, 3)
    train_datapath = "./stage1_train/"
    test_datapath  = "./stage1_test/"
    dataset = NucleiDataset(resized_image)
    x_train, y_train = dataset.create(train_datapath, has_mask=True)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    x_test, y_test   = dataset.create(train_datapath, has_mask=False)
    print(" len x_test {}".format(len(x_test)))


  except:
    traceback.print_exc()

