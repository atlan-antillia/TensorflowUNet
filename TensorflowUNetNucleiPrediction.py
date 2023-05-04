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

# TensorflowUNetNucleiTester.py
# 2023/05/05 to-arai


import os
import shutil

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import traceback

from ConfigParser import ConfigParser
from NucleiDataset import NucleiDataset
from EpochChangeCallback import EpochChangeCallback

from TensorflowUNet import TensorflowUNet
from GrayScaleImageWriter import GrayScaleImageWriter

MODEL  = "model"
TRAIN  = "train"
TEST   = "test"


if __name__ == "__main__":
  try:
    config_file    = "./predict.config"
    config     = ConfigParser(config_file)

    width      = config.get(MODEL, "image_width")
    height     = config.get(MODEL, "image_height")
    channels   = config.get(MODEL, "image_channels")
    output_dir = config.get(TEST,  "output_dir")
    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model          = TensorflowUNet(config_file)
    
    test_datapath = "./stage1_test/" 

    if not os.path.exists(test_datapath):
      raise Exception("Not found " + test_datapath)

    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    resized_image    = (height, width, channels)
    dataset          = NucleiDataset(resized_image)
    x_test, y_test = dataset.create(test_datapath, has_mask=False)
    print("x_test len {}".format(len(x_test)) )
   
    writer = GrayScaleImageWriter()

    predictions = model.predict(x_test, expand=True)
    n = 101
    for i, prediction in enumerate(predictions):
      image       = prediction[0]    
      writer.save(image, output_dir, "pred_test_" + str(n + i))

  except:
    traceback.print_exc()
    
