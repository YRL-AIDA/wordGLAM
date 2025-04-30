import subprocess

def start_new_exp(name, header="exp_03_countTag"):
    with open ('.env', 'r') as f:
        old_data = f.read()
    new_data = "\n".join([f'EXPERIMENT="{header}/{name}"']+old_data.split("\n")[1:])
    with open ('.env', 'w') as f:
        f.write(new_data)
    

    subprocess.run(["python", "script_train.py"])
    subprocess.run(["python", "script_test.py"])


for coef in ['01_2', '02_3', '03_4']:
    start_new_exp(coef)
