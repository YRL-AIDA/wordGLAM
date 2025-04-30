import subprocess

def start_new_exp(name, header="exp_04_kTag"):
    with open ('.env', 'r') as f:
        old_data = f.read()
    new_data = "\n".join([f'EXPERIMENT="{header}/{name}"']+old_data.split("\n")[1:])
    with open ('.env', 'w') as f:
        f.write(new_data)
    

    subprocess.run(["python", "script_train.py"])
    subprocess.run(["python", "script_test.py"])


for coef in ['01_1', '02_2', '03_3', '04_4', '05_5', '06_6']:
    start_new_exp(coef)
