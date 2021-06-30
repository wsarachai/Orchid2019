from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.mobilenet_v2_140 import create_mobilenet_v2_14
from nets.mobilenet_v2_140 import preprocess_input
from nets.mobilenet_v2_140_orchids52 import create_orchid_mobilenet_v2_15

TRAIN_TEMPLATE = "pretrain{}"
TRAIN_STEP1 = "pretrain1"
TRAIN_STEP2 = "pretrain2"
TRAIN_STEP3 = "pretrain3"
TRAIN_STEP4 = "pretrain4"
TRAIN_V2_STEP1 = "v2-pretrain1"
TRAIN_V2_STEP2 = "v2-pretrain2"

MOBILENET_V2_140 = "mobilenet_v2_140"
MOBILENET_V2_140_FLOWERS17 = "mobilenet_v2_140_flowers17"
MOBILENET_V2_140_FLOWERS102 = "mobilenet_v2_140_flowers102"
MOBILENET_V2_140_ORCHIDS52 = "mobilenet_v2_140_stn_v15"


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
