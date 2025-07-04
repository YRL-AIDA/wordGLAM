import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS["node_featch"]= 15
EXPERIMENT_PARAMS["loss_params"]['edge_coef'] =0.8
EXPERIMENT_PARAMS["loss_params"]['node_coef'] =0.2

EXPERIMENT_PARAMS["Tag"] = [{"in":-1, "size": 1024, "out":512, "k": 3}, 
                            {"in":512, "size": 256, "out":128, "k": 3},
                            {"in":128, "size": 64,  "out":32, "k": 3},
                            {"in":32, "size": 16, "out":8, "k":3}]