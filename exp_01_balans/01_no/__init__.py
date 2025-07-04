import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS["loss_params"]["publaynet_imbalance"] = [1.0, 1.0, 1.0, 1.0,1.0]
EXPERIMENT_PARAMS["loss_params"]["edge_imbalance"] = 1

