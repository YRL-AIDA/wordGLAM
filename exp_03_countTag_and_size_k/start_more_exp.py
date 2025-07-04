import subprocess
import os

HEADER = "exp_03_countTag_and_size_k"

sizes = [
    [[-1, 64, 16]],
    [[-1, 64, 64], [64, 16, 16]],
    [[-1, 64, 64], [64, 32, 32], [32, 16, 16]],
    [[-1, 64, 64], [64, 48, 48], [48, 24, 24], [24, 16, 16]],
    [[-1, 64, 64], [64, 48, 48], [48, 32, 32], [32, 24, 24], [24, 16, 16]],
]

def created_exps():
    
    for i in range(1, 6):
        for j in range(2, 7):
            dir_ = os.path.join(HEADER, f"exp_{i}_kTag_{j}")
            os.mkdir(dir_)
            file_ = os.path.join(dir_, "__init__.py")
            exp_str = [f'{{ "in": {size[0]}, "size": {size[1]}, "out": {size[2]}, "k":{j}}}' for size in sizes[i-1]]
            exp_str = ",\n".join(exp_str) 
            with open(file_, 'w') as f:
                f.write(f"""
import sys, os
sys.path.append("..")
from exp_00_base import *

EXPERIMENT_PARAMS["Tag"] = [{exp_str}]
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



    

if not os.path.exists(os.path.join(HEADER, "exp_2_kTag_2")):
    created_exps()
for i in range(1, 6):
    for j in range(2, 7):
        dir_ = os.path.join(HEADER, f"exp_{i}_kTag_{j}", "test_result.txt")
        if not os.path.exists(dir_):
            start_new_exp(f"exp_{i}_kTag_{j}")