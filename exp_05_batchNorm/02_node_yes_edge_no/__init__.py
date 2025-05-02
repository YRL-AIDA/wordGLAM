import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS["batchNormNode"] = True
EXPERIMENT_PARAMS["batchNormEdge"] = False