
import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS["Tag"] = [{ "in": -1, "size": 64, "out": 64, "k":2},
{ "in": 64, "size": 32, "out": 32, "k":2}]
