from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
from lib_utils import start


def printname(name):
    print(name)


def main(unused_argv):
    filepath = '/home/keng/.keras/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.4_224_no_top.h5'
    with h5py.File(filepath, 'r') as f:
        f.visit(printname)


if __name__ == '__main__':
    start(main)