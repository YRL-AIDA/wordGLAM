import subprocess

def start_new_exp(name, header="exp_02_complex_loss"):
    with open ('.env', 'r') as f:
        old_data = f.read()
    new_data = "\n".join([f'EXPERIMENT="{header}/{name}"']+old_data.split("\n")[1:])
    with open ('.env', 'w') as f:
        f.write(new_data)
    

    subprocess.run(["python", "script_train.py"])
    subprocess.run(["python", "script_test.py"])


for coef in ['01_005', '02_025', '03_050', '04_075', '05_085', '06_095', '07_100']:
    start_new_exp(coef)
