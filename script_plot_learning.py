import matplotlib.pyplot as plt
from config import LOG_FILE

# LOG_FILE = "exp_04_kTag/01_1/log.txt"
# LOG_FILE = "exp_04_kTag/02_2/log.txt"
# LOG_FILE = "exp_04_kTag/03_3/log.txt"
# LOG_FILE = "exp_04_kTag/04_4/log.txt"
# LOG_FILE = "exp_04_kTag/05_5/log.txt"
# LOG_FILE = "exp_04_kTag/06_6/log.txt"

# LOG_FILE = "exp_02_complex_loss/01_005/log.txt"
# LOG_FILE = "exp_02_complex_loss/02_025/log.txt"
# LOG_FILE = "exp_02_complex_loss/03_050/log.txt"
# LOG_FILE = "exp_02_complex_loss/04_075/log.txt"
# LOG_FILE = "exp_02_complex_loss/05_095/log.txt"

# LOG_FILE = "exp_05_batchNorm/01_node_no_edge_no/log.txt"
# LOG_FILE = "exp_05_batchNorm/02_node_yes_edge_no/log.txt"
# LOG_FILE = "exp_05_batchNorm/03_node_no_edge_yes/log.txt"
# LOG_FILE = "exp_05_batchNorm/04_node_yes_edge_yes/log.txt"

# LOG_FILE = "exp_d01_restart/01/log.txt"
# LOG_FILE = "exp_d01_restart/02/log.txt"
# LOG_FILE = "exp_d01_restart/03/log.txt"

if __name__ == '__main__':
    with open(LOG_FILE, 'r') as f:
        text = f.read()
        rez = text.split("DATASET INFO:")[-1]
        lines = [l for l in rez.split('\n') if len(l) >5 and l[:5]=="EPOCH"]
        
        # lines = f.readlines()
        
        train = [ float(line.split(' ')[2]) for line in lines]
        val = [ float(line.split('VAL: ')[1][:-2]) for line in lines]
    plt.plot(train, label='train')
    plt.plot(val, label='val')
    plt.xlabel('epoch')
    plt.legend()
    plt.grid()
    plt.show()