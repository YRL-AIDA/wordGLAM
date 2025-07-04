
import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS["Tag"] = [{ "in": -1, "size": 64, "out": 64, "k":6},
{ "in": 64, "size": 32, "out": 32, "k":6},
{ "in": 32, "size": 16, "out": 16, "k":6}]
