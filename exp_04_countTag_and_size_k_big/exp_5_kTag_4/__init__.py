
import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS["Tag"] = [{ "in": -1, "size": 512, "out": 512, "k":4},
{ "in": 512, "size": 256, "out": 256, "k":4},
{ "in": 256, "size": 128, "out": 128, "k":4},
{ "in": 128, "size": 64, "out": 64, "k":4},
{ "in": 64, "size": 32, "out": 32, "k":4}]
