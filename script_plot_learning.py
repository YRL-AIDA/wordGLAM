import matplotlib.pyplot as plt
from config import LOG_FILE


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