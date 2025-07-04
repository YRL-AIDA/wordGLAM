import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS["loss_params"]["publaynet_imbalance"] = [2.63, 0.015, 0.946, 1.268, 0.136]
EXPERIMENT_PARAMS["loss_params"]["edge_imbalance"] = 0.14