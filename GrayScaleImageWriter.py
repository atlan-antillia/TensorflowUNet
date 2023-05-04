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

# GrayImageWriter.py

# 2023/05/05 to-arai

import os

from PIL import Image

class GrayScaleImageWriter:

  def __init__(self, image_format=".jpg"):
    self.image_format = image_format

  def save(self, data, output_dir, name):
    
    (h, w, c) = data.shape
    image = Image.new("L", (w, h))
    for i in range(h):
      for j in range(w):
        z = data[i][j]
        v = int(z[0]*255.0)
        image.putpixel((i,j), v)
    if not os.path.exists(output_dir):
      os.makedirs(outout_dir)

    image_filepath = os.path.join(output_dir, name + self.image_format)

    image.save(image_filepath)
    print("=== Saved {}". format(image_filepath))