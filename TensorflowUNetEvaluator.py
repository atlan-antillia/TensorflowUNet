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

# TensorflowUNetEvaluator.py
# 2023/05/05 to-arai
# 2023/06/17 Modified filename to TensorflowUNetEvaluator.py

import os
import sys
import shutil

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset

from TensorflowUNet import TensorflowUNet
from GrayScaleImageWriter import GrayScaleImageWriter

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"


if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]
    if not os.path.exists(config_file):
      raise Exception("Not found " + config_file)
    
    config     = ConfigParser(config_file)

    width      = config.get(MODEL, "image_width")
    height     = config.get(MODEL, "image_height")
    output_dir = config.get(EVAL,  "output_dir")
    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model          = TensorflowUNet(config_file)
  
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # 1 Create ImageMaskDataset
    dataset          = ImageMaskDataset(config_file)
    x_test, y_test = dataset.create(dataset=EVAL)

    print("x_test len {}".format(len(x_test)) )

    # 2 Create a UNetMolde and compile
    model          = TensorflowUNet(config_file)

    # 3 Train the model by train dataset
    model.evaluate(x_test, y_test)    
   
  except:
    traceback.print_exc()
    
