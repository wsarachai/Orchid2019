from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.core_functions import preprocess_input
from nets.mobilenet_v2 import create_mobilenet_v2_14
from nets.v1.mobilenet_v2_140_orchids52 import create_orchid_mobilenet_v2_15 as create_orchid_mobilenet_v2_15_v1
from nets.mobilenet_v2_140_orchids52 import create_orchid_mobilenet_v2_15
from nets.resnet_v2_50_orchids52 import create_orchid_resnet_v2_50_15
from utils.const import MOBILENET_V2_140_V1, MOBILENET_V2_140
from utils.const import MOBILENET_V2_140_FLOWERS17_V1, MOBILENET_V2_140_FLOWERS17
from utils.const import MOBILENET_V2_140_FLOWERS102_V1, MOBILENET_V2_140_FLOWERS102
from utils.const import MOBILENET_V2_140_ORCHIDS52_V1, MOBILENET_V2_140_ORCHIDS52
from utils.const import RESNET_V2_50_ORCHIDS52
from utils.wrapped_tools import wrapped_partial

create_mobilenet_v2_14_v1 = wrapped_partial(create_mobilenet_v2_14, ver=1)
create_mobilenet_v2_14_v2 = wrapped_partial(create_mobilenet_v2_14, ver=2)

nets_mapping = {
    MOBILENET_V2_140_V1: create_mobilenet_v2_14_v1,
    MOBILENET_V2_140_FLOWERS17_V1: create_orchid_mobilenet_v2_15_v1,
    MOBILENET_V2_140_FLOWERS102_V1: create_orchid_mobilenet_v2_15_v1,
    MOBILENET_V2_140_ORCHIDS52_V1: create_orchid_mobilenet_v2_15_v1,
    MOBILENET_V2_140: create_mobilenet_v2_14_v2,
    MOBILENET_V2_140_FLOWERS17: create_orchid_mobilenet_v2_15,
    MOBILENET_V2_140_FLOWERS102: create_orchid_mobilenet_v2_15,
    MOBILENET_V2_140_ORCHIDS52: create_orchid_mobilenet_v2_15,
    RESNET_V2_50_ORCHIDS52: create_orchid_resnet_v2_50_15,
}

preprocessing_mapping = {
    MOBILENET_V2_140_V1: preprocess_input,
    MOBILENET_V2_140_FLOWERS17_V1: preprocess_input,
    MOBILENET_V2_140_FLOWERS102_V1: preprocess_input,
    MOBILENET_V2_140_ORCHIDS52_V1: preprocess_input,
    MOBILENET_V2_140: preprocess_input,
    MOBILENET_V2_140_FLOWERS17: preprocess_input,
    MOBILENET_V2_140_FLOWERS102: preprocess_input,
    MOBILENET_V2_140_ORCHIDS52: preprocess_input,
}
