import subprocess

def start_new_exp(name, header="exp_05_batchNorm"):
    with open ('.env', 'r') as f:
        old_data = f.read()
    new_data = "\n".join([f'EXPERIMENT="{header}/{name}"']+old_data.split("\n")[1:])
    with open ('.env', 'w') as f:
        f.write(new_data)
    

    subprocess.run(["python", "script_train.py"])
    subprocess.run(["python", "script_test.py"])


for coef in ['01_node_no_edge_no', '02_node_yes_edge_no', '03_node_no_edge_yes', '04_node_yes_edge_yes']:
    start_new_exp(coef)
