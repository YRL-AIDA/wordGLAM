import subprocess

def start_new_exp(name):
    subprocess.run(["python", f"{name}/start_more_exp.py"])

# start_new_exp("exp_01_balans")
start_new_exp("exp_02_complex_loss")
start_new_exp("exp_03_countTag")
start_new_exp("exp_04_kTag")
# start_new_exp("exp_05_batchNorm")