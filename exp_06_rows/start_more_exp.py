import subprocess
import os

HEADER = "exp_06_rows"

sizes = [
    [0.01, 0.19, 0.8],
    [0.02, 0.18, 0.8],
    [0.04, 0.16, 0.8],
    [0.1, 0.1, 0.8],
    [0.16, 0.04, 0.8],
    [0.18, 0.02, 0.8],
    [0.19, 0.01, 0.8],
]

def created_exps():
    
    for i in range(0, len(sizes)):
        dir_ = os.path.join(HEADER, f"exp_{i}")
        os.mkdir(dir_)
        file_ = os.path.join(dir_, "__init__.py")
        with open(file_, 'w') as f:
            f.write(f"""
import sys, os
sys.path.append("..")
from exp_06_rows import *

EXPERIMENT_PARAMS["loss_params"]["edge_coef"] = {sizes[i][0]}
EXPERIMENT_PARAMS["loss_params"]["row_coef"] = {sizes[i][1]}
EXPERIMENT_PARAMS["loss_params"]["node_coef"] = {sizes[i][2]}
"""
)
def start_new_exp(name, header=HEADER):
    with open ('.env', 'r') as f:
        old_data = f.read()
    new_data = "\n".join([f'EXPERIMENT="{header}/{name}"']+old_data.split("\n")[1:])
    with open ('.env', 'w') as f:
        f.write(new_data)
    

    subprocess.run(["python", "script_train.py"])
    subprocess.run(["python", "script_test.py"])


if not os.path.exists(os.path.join(HEADER, "exp_0")):
    created_exps()
for i in range(0, len(sizes)):
    dir_ = os.path.join(HEADER, f"exp_{i}", "test_result.txt")
    if not os.path.exists(dir_):
        start_new_exp(f"exp_{i}")