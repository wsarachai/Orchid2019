from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.constants import MOBILENET_V2_140, MOBILENET_V2_140_ORCHIDS52
from nets.mobilenet_v2_140 import create_mobilenet_v2_14_v1
from nets.mobilenet_v2_140_orchids52 import create_orchid_mobilenet_v2_14

nets_mapping = {
    MOBILENET_V2_140: create_mobilenet_v2_14_v1,
    MOBILENET_V2_140_ORCHIDS52: create_orchid_mobilenet_v2_14
}
