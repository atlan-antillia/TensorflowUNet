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
from PIL import Image
import cv2
import numpy as np
import shutil
from matplotlib import pyplot as plt

# pip install scikit-image
#from skimage.transform import resize
#from skimage.morphology import label
#from skimage.io import imread, imshow, imsave
import glob
import traceback

MODEL = "model"
TRAIN = "train"
EVAL  = "eval"

# Input dataset
"""
./stage1_train
...
├─a9d884ba0929dac87c2052ce5b15034163685317d7cff45c40b0f7bd9bd4d9e7
  ├─images
  ├─ └─a9d884ba0929dac87c2052ce5b15034163685317d7cff45c40b0f7bd9bd4d9e7.png
  ├─masks
    ├─a0f33d2a4910a6552ef6c2b8e302bf0d58c6c2efa802d6f1b70dafcdc6969503d.png
    ├─a01c32faf1b4b286415cae803cbd9aaa10bb5495352b3428dbdaf4f677597a2b9.png
    ....
    └─1b137dfad04c56041bc038deff17c56523ea82ccbac474d2e63a7b0a1675d5c6.png
     
"""

# Output dataset
"""
Nuclei
 +-- train
 |     +-- images
 |     |    +-- a9d884ba0929dac87c2052ce5b15034163685317d7cff45c40b0f7bd9bd4d9e7.png
 ...
 |     +-- masks
 |          +-- a9d884ba0929dac87c2052ce5b15034163685317d7cff45c40b0f7bd9bd4d9e7.png
 ...
 +-- valid
       +-- images
       |    +-- 1b137dfad04c56041bc038deff17c56523ea82ccbac474d2e63a7b0a1675d5c6.png
       ...
       +-- masks
            +--  1b137dfad04c56041bc038deff17c56523ea82ccbac474d2e63a7b0a1675d5c6.png
            ...

"""

class NucleiDatasetPreprocessor:

  def __init__(self, resized_image):
    
    self.IMG_WIDTH, self.IMG_HEIGHT, self.IMG_CHANNELS  = resized_image 
    
  #  train_datapath = "./stage1_train/id/images/id.png"
  #  
  def create(self, data_dir, output_dir, split=True):
    image_files = sorted(glob.glob(data_dir + "/*/images/*.png"))
    if split:
      num_files   = len(image_files)
      num_train   = int(num_files * 0.8) 
      num_valid   = int(num_files * 0.2)
      train_files = image_files[0: num_train]
      valid_files = image_files[num_train: num_train+num_valid]
      print("--- num_files {}".format(num_files))
      print("--- num_train {}".format(num_train))
      print("--- num_valid {}".format(num_valid))
    
      self.create_dataset(data_dir, train_files, output_dir, dataset="train")
      self.create_dataset(data_dir, valid_files, output_dir, dataset="valid")
    else:
      self.create_dataset(data_dir, image_files, output_dir, dataset="test")


  def create_dataset(self, data_dir, image_files, output_dir, dataset="train"):
    dataset_dir = os.path.join(output_dir, dataset)

    # Create images and masks dataset
    # 
    images_dir = os.path.join(dataset_dir, "images")
    
    masks_dir  = os.path.join(dataset_dir, "masks")

    if  os.path.exists(images_dir):
      shutil.rmtree(images_dir)
    if not os.path.exists(images_dir):
      os.makedirs(images_dir)

    if  os.path.exists(masks_dir):
      shutil.rmtree(masks_dir)
    if not os.path.exists(masks_dir):
      os.makedirs(masks_dir)

    for image_file in image_files:
      print("=== image_file {}".format(image_file))
    
      img = cv2.imread(image_file)
      img = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))

      image_file = image_file.replace("\\", "/")
      paths     = image_file.split("/")
      image_id  = paths[2]
      output_imagefile = image_id + ".png"
      output_image_filepath = os.path.join(images_dir, output_imagefile)
      cv2.imwrite(output_image_filepath, img)
      print("--- Saved image file {}".format(output_image_filepath))

      mask_files = glob.glob(data_dir + "/" + image_id + "/masks/*.png")
      if len(mask_files) >0:
        mask = np.zeros((self.IMG_HEIGHT, self.IMG_WIDTH, 3), np.uint8)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        for mask_file in mask_files:
          mask_ = cv2.imread(mask_file)
          mask_ = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
          mask_ = cv2.resize(mask_, (self.IMG_WIDTH, self.IMG_HEIGHT))

          mask = np.maximum(mask, mask_)
        
        output_maskfile = image_id + ".png"
        output_mask_filepath = os.path.join(masks_dir, output_maskfile)
        cv2.imwrite(output_mask_filepath, mask)
        print("--- Saved mask file {}".format(output_mask_filepath))


if __name__ == "__main__":
 
  try:

    resized_image = (256, 256, 3)
    train_datapath   = "./stage1_train/"
    test_datapath    = "./stage1_test/"

    output_dir = "./Nuclei/"
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    dataset = NucleiDatasetPreprocessor(resized_image)
   
    dataset.create(train_datapath, output_dir, split=True)

    dataset.create(test_datapath,  output_dir, split=False)
    

  except:
    traceback.print_exc()

