
import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS["Tag"] = [{ "in": -1, "size": 64, "out": 64, "k":5},
{ "in": 64, "size": 48, "out": 48, "k":5},
{ "in": 48, "size": 32, "out": 32, "k":5},
{ "in": 32, "size": 24, "out": 24, "k":5},
{ "in": 24, "size": 16, "out": 16, "k":5}]
