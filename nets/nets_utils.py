from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nets.mobilenet_v2_140 import create_mobilenet_v2_14
from nets.mobilenet_v2_140_orchids52 import create_orchid_mobilenet_v2_14

MOBILENET_V2_140 = 'const/MOBILENET_V2_140'
MOBILENET_V2_140_ORCHIDS52 = 'const/MOBILENET_V2_140-ORCHiDS52'


nets_mapping = {
    MOBILENET_V2_140: create_mobilenet_v2_14,
    MOBILENET_V2_140_ORCHIDS52: create_orchid_mobilenet_v2_14
}
