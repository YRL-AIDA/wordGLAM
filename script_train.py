import torch
from torch.nn import Linear, BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, GELU
from torch.nn.functional import relu
from torch_geometric.nn import BatchNorm, TAGConv
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import os
import json
import time
from config import GLAM_MODEL, LOG_FILE, PARAMS,SAVE_FREQUENCY,PATH_GRAPHS_JSONS
from config import TorchModel, CustomLoss
device = torch.device('cuda:0' if torch.cuda.device_count() != 0 else 'cpu')
SKIP_INDEX = []
device = torch.device('cpu')
torch.manual_seed(seed=1234)
np.random.seed(1234)

class GLAMDataset(Dataset):
    def __init__(self, json_dir):
        self.json_dir = json_dir
        files  = sorted(os.listdir(self.json_dir))

        if os.path.exists("error_list_file.txt"):
            with open("error_list_file.txt", "r") as f:
                lines = f.readlines()
            error_file = [int(line.split(" ")[0]) for line in lines]
        else:       
            print("TEST OPEN FILE:")
            json_error = []
            key_error = []
            N = len(files)
            for i, file in enumerate(files):
                print(f"{i+1}/{N} ({(i+1)/N*100:.2f} %)" + " "*10, end="\r")
                try:
                    path = os.path.join(self.json_dir, file)
                    with open(path, "r") as f: 
                        j = json.load(f)
                    for k in ["nodes_feature", "edges_feature", "true_edges", "true_nodes"]:                    
                        if not k in j:
                            key_error.append(i)
                            raise KeyError(f"{k} not in {file}")
                except:
                    json_error.append(i)
            
            if len(key_error) != 0:
                print("KEY ERROR FILES:")
                log("KEY ERROR FILES:")
                for i in key_error:
                    print(files[i])
                    log(files[i])

            if len(json_error) != 0:
                print("JSON ERROR FILES:")
                log("JSON ERROR FILES:")
                for i in json_error:
                    print(files[i])
                    log(files[i])
            error_file = sorted(key_error + json_error, reverse=True)
            with open("error_list_file.txt", "w") as f:
                for i in error_file:
                    f.write(str(i) + " "+ files[i] + '\n')
        for i in error_file:
            del files[i]
        self.files = files
        self.count = len(self.files)

        

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        path = os.path.join(self.json_dir, self.files[idx])
        with open(path, 'r') as f:
            data = json.load(f)
        return data
    
    def __str__(self):
        return f"""DATASET INFO:
count row: {len(self)}
first: {self[0].keys()}
\t A:{np.shape(self[0]["A"])}
\t nodes_feature:{np.shape(self[0]["nodes_feature"])}
\t edges_feature:{np.shape(self[0]["edges_feature"])}
\t true_edges:{np.shape(self[0]["true_edges"])}
\t true_nodes:{np.shape(self[0]["true_nodes"])}
end:{self[-1].keys()}
\t A{np.shape(self[-1]["A"])}
\t nodes_feature:{np.shape(self[-1]["nodes_feature"])}
\t edges_feature:{np.shape(self[-1]["edges_feature"])}
\t true_edges:"{np.shape(self[-1]["true_edges"])}
\t true_nodes:{np.shape(self[-1]["true_nodes"])}

"""

def delete_error_nodes(graph):
    error_nodes = [i for i, n in enumerate(graph["true_nodes"]) if n == -1]
    true_nodes = [i for i, n in enumerate(graph["true_nodes"]) if n != -1]
    for index in sorted(error_nodes, reverse=True):
        del graph["nodes_feature"][index]
        del graph["true_nodes"][index]

    error_edges = [i for i, e in enumerate(zip(graph["A"][0], graph["A"][1])) 
                   if e[0] in error_nodes or 
                      e[1] in error_nodes]
  
    for index in sorted(error_edges, reverse=True):
        del graph["A"][0][index]
        del graph["A"][1][index]
        del graph["edges_feature"][index]
        del graph["true_edges"][index]
    new = dict()
    for i, n in enumerate(true_nodes):
        new[n] = i
    
    for i in range(len(graph["A"][0])):
        graph["A"][0][i] = new[graph["A"][0][i]]
        graph["A"][1][i] = new[graph["A"][1][i]]
        

def get_tensor_from_graph(graph):
    def class_node(n):
        rez = [0, 0, 0, 0, 0]
        if n!= -1:
            rez[n] = 1
        return rez
    delete_error_nodes(graph)
    i = graph["A"]
    # v_in = [1 for e in graph["edges_feature"]]
    y = graph["edges_feature"]

    v_true = graph["true_edges"]
    n_true = [class_node(n) for n in graph["true_nodes"]]
    x = graph["nodes_feature"]
    N = len(x)
    
    X = torch.tensor(data=x, dtype=torch.float32).to(device)
    Y = torch.tensor(data=y, dtype=torch.float32).to(device)
    j_down = [[i1, i0] for i0, i1 in zip(i[0], i[1]) if i0 != i1]
    sp_indices = [i[0] + [j[0] for j in j_down ],
                 i[1] + [j[1] for j in j_down ]]
    sp_values = [1 for e in sp_indices[0]]
    sp_A = torch.sparse_coo_tensor(indices=sp_indices, values=sp_values, size=(N, N), dtype=torch.float32).to(device)
    E_true = torch.tensor(data=v_true, dtype=torch.float32).to(device)
    N_true = torch.tensor(data=n_true, dtype=torch.float32).to(device)
    if len(X.shape) != 2 or X.shape[1] != PARAMS["node_featch"]:
        X = []
    if len(Y.shape) != 2 or Y.shape[1] != PARAMS["edge_featch"] or len(Y[0]) in (0, 1):
        X = []
    return X, Y, sp_A, E_true, N_true, i

def validation(model, batch, criterion):     
    return step(model, batch, optimizer=None, criterion=criterion, train=False)


def split_index_train_val(dataset, val_split=0.2, shuffle=True, seed=1234,batch_size=64):
    N = len(dataset)
    count_batchs = int(N*(1-val_split))//batch_size
    count_val_batch = int(N*(val_split))//batch_size
    train_size = count_batchs * batch_size 
    indexs = [i for i in range(N)]
    if shuffle:
        np.random.shuffle(indexs)
    train_indexs = indexs[:train_size]
    val_indexs = indexs[train_size:]
    batchs_train_indexs = [[train_indexs[k*batch_size+i] for i in range(batch_size)] for k in range(count_batchs)]
    batchs_val_indexs = [[val_indexs[k*batch_size+i] for i in range(batch_size)] for k in range(count_val_batch)]
    return batchs_train_indexs, batchs_val_indexs    

def step(model: torch.nn.Module, batch, optimizer, criterion, train=True):
    if train:
        optimizer.zero_grad()
    my_loss_list = []
   
    for j, graph in enumerate(batch):
        try:
            X, Y, sp_A, E_true, N_true, i = get_tensor_from_graph(graph)
            if len(X) in (0, 1):                       
                continue
            Node_class, E_pred = model(X, Y, sp_A, i)
            loss = criterion(Node_class.to(device), N_true, E_pred.to(device), E_true)
            my_loss_list.append(loss.item())
            print(f"{(j+1)/len(batch)*100:.2f} % Batch loss={my_loss_list[-1]:.4f}" + " "*40, end="\r")
        except Exception as e:
            print(e)
            if "edges_feature" in graph.keys():
                print(np.array(graph['edges_feature']).shape)
            if "nodes_feature" in graph.keys():
                print(np.array(graph['nodes_feature']).shape)
            continue
        if train:  
            loss.backward()
    if train:
        optimizer.step()
    return np.mean(my_loss_list)

def train_model(params, model, dataset, save_frequency=5, start_epoch=0):  
    optimizer = torch.optim.Adam(
    list(model.parameters()),
    lr=params["learning_rate"],
    )
    criterion = CustomLoss(params["loss_params"]) 

    model.to(device)
    criterion.to(device)

    loss_list = []
    start = time.time()
    train_dataset, val_dataset = split_index_train_val(dataset, val_split=0.1, batch_size=params["batch_size"])
    for k in range(start_epoch, params["epochs"]):
        my_loss_list = []
        if k == start_epoch:
            start = time.time()
        for l, batch_indexs in enumerate(train_dataset):
            batch = [dataset[ind] for ind in batch_indexs]
            batch_loss = step(model, batch, optimizer, criterion)
            my_loss_list.append(batch_loss)
            print(f"Batch # {l+1} loss={my_loss_list[-1]:.4f}" + " "*40)
            if (k == start_epoch and l==0):
                print(f"Время обучения batch'а {time.time()-start:.2f} сек")
        train_val = np.mean(my_loss_list)
        loss_list.append(train_val)

        my_loss_list = []
        for l, batch_indexs in enumerate(val_dataset):
            batch = [dataset[ind] for ind in batch_indexs]
            batch_loss = validation(model, batch, criterion)
            my_loss_list.append(batch_loss)
            print(f"Batch # {l+1} loss={my_loss_list[-1]:.4f}" + " "*40)
        validation_val =  np.mean(my_loss_list)
        print("="*10, f"EPOCH #{k+1}","="*10, f"({train_val:.4f}/{validation_val:.4f})")
        if k == start_epoch:
            print(f"Время обучения epoch {time.time()-start:.2f} сек")    
            
        log(f"EPOCH #{k}\t {train_val:.8f} (VAL: {validation_val:.8f})")  
        if (k+1) % save_frequency == 0:
            num = k//save_frequency
            torch.save(model.state_dict(), GLAM_MODEL+f"_tmp_{num}")
    log(f"Время обучения: {time.time()-start:.2f} сек")
    torch.save(model.state_dict(), GLAM_MODEL)


def load_checkpoint(model, path_model,restart_num=None):
    dir_model = os.path.dirname(path_model)
    name_model = os.path.basename(path_model)
    names = [n for n in os.listdir(dir_model) if name_model+'_tmp_' in n]
    if restart_num is None:
        list_num = [int(n.split("_tmp_")[-1]) for n in names]
        if len(list_num) == 0:
            return
        restart_num = max(list_num) 

    checkpoint_path = os.path.join(dir_model, name_model+f"_tmp_{restart_num}")
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    print(checkpoint_path)
    return restart_num

def log(str_):
    with open(LOG_FILE, 'a') as f:
        f.write(str_+'\n')

if __name__ == "__main__":
    is_restart = False
    restart_num = None
    dataset = GLAMDataset(PATH_GRAPHS_JSONS)
    import datetime
    if is_restart:
        log("R E S T A R T ")
    log(datetime.datetime.now().__str__())
    try:
        str_ = dataset.__str__()
        str_ += '\n'.join(f"{key}:\t{val}" for key, val in PARAMS.items())
        print(str_)
        if not is_restart:
            log(str_)
    except:
        print(dataset)

    model:torch.nn.Module = TorchModel(PARAMS)
    if is_restart:
        restart_num = load_checkpoint(model, GLAM_MODEL)
    
    start_epoch = 0 if restart_num is None else (restart_num+1)*SAVE_FREQUENCY
    train_model(PARAMS, model, dataset, save_frequency=SAVE_FREQUENCY, start_epoch=start_epoch)