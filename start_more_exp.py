import subprocess

def start_new_exp(name):
    with open ('.env', 'r') as f:
        old_data = f.read()
    new_data = "\n".join([f'EXPERIMENT="{name}"']+old_data.split("\n")[1:])
    with open ('.env', 'w') as f:
        f.write(new_data)
    

    subprocess.run(["python", "script_train.py"])


start_new_exp("exp_02_3convTag")
start_new_exp("exp_03_4convTag")