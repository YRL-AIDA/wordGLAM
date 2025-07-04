import subprocess

def start_new_exp(name, header="exp_hard_01_wordsize512"):
    with open ('.env', 'r') as f:
        old_data = f.read()
    new_data = "\n".join([f'EXPERIMENT="{header}/{name}"']+old_data.split("\n")[1:])
    with open ('.env', 'w') as f:
        f.write(new_data)
    

    subprocess.run(["python", "script_train.py"])
    subprocess.run(["python", "script_test.py"])

start_new_exp("01_no")
start_new_exp("02_yes")
start_new_exp("03_heuristics")
