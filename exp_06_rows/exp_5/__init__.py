
import sys, os
sys.path.append("..")
from exp_06_rows import *

EXPERIMENT_PARAMS["loss_params"]["edge_coef"] = 0.6
EXPERIMENT_PARAMS["loss_params"]["row_coef"] = 0.2
EXPERIMENT_PARAMS["loss_params"]["node_coef"] = 0.2
