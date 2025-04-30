import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS = {
    "node_featch": 47,
    "edge_featch": 2,
    "epochs": 30,
    "batch_size": 80,
    "learning_rate": 0.005,
    "Tag": [{"in":-1, "size": 256, "out":128, "k":3}, 
            {"in":128, "size": 128, "out":128, "k":3},
            {"in":128, "size": 64, "out":32, "k":3}],
    "NodeLinear": [-1, 16, 8],
    "EdgeLinear": [8],
    "NodeClasses":5,
    "seg_k": 0.5
}