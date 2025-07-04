
import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS["Tag"] = [{ "in": -1, "size": 256, "out": 256, "k":3},
{ "in": 256, "size": 128, "out": 128, "k":3},
{ "in": 128, "size": 64, "out": 64, "k":3},
{ "in": 64, "size": 32, "out": 32, "k":3}]
