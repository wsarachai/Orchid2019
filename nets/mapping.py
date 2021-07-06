from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.mobilenet_v2_140 import create_mobilenet_v2_14
from nets.mobilenet_v2_140 import preprocess_input
from nets.mobilenet_v2_140_orchids52 import create_orchid_mobilenet_v2_15
from utils.const import MOBILENET_V2_140
from utils.const import MOBILENET_V2_140_FLOWERS17
from utils.const import MOBILENET_V2_140_FLOWERS102
from utils.const import MOBILENET_V2_140_ORCHIDS52

nets_mapping = {
    MOBILENET_V2_140: create_mobilenet_v2_14,
    MOBILENET_V2_140_FLOWERS17: create_orchid_mobilenet_v2_15,
    MOBILENET_V2_140_FLOWERS102: create_orchid_mobilenet_v2_15,
    MOBILENET_V2_140_ORCHIDS52: create_orchid_mobilenet_v2_15,
}

preprocessing_mapping = {
    MOBILENET_V2_140: preprocess_input,
    MOBILENET_V2_140_FLOWERS17: preprocess_input,
    MOBILENET_V2_140_FLOWERS102: preprocess_input,
    MOBILENET_V2_140_ORCHIDS52: preprocess_input,
}
