from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                    'keras-applications/mobilenet_v2/')
default_image_size = 224
IMG_SIZE_224 = (default_image_size, default_image_size)
IMG_SHAPE_224 = IMG_SIZE_224 + (3,)
regularizers_l2 = 0.00004
