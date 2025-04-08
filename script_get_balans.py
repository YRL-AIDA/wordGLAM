import os
import json
from config import PATH_GRAPHS_JSONS


def count_elem_class(path, list_class, list_class2, key, key2):
    graphs = os.listdir(path)
    counts = [0 for _ in list_class]
    counts2 = [0 for _ in list_class2]
    N = len(graphs)
    for i, gr in enumerate(graphs):
        print(f"{i+1}/{N} {(i+1)/N*100:.2f} %" + " "*10, end="\r")
        graph_path = os.path.join(path, gr)
        try:
            with open(graph_path, "r") as f:
                graph = json.load(f)
                for i, cl in enumerate(list_class):
                    r = graph[key].count(cl)
                    counts[i] += r
                for i, cl in enumerate(list_class2):
                    r = graph[key2].count(cl)
                    counts2[i] += r
        except:
            pass
    return counts, counts2

def get_balans(path, list_class,list_class2, key, key2):
    counts, counts2 = count_elem_class(path,list_class,list_class2, key, key2)
    mx = max(counts)
    mx2 = max(counts2)
    balans = [mx/c for c in counts]
    balans2 = [mx2/c for c in counts2]
    coef = len(list_class)/sum(balans)
    coef2 = len(list_class2)/sum(balans2)
    balans_norm = [coef*b for b in balans]
    balans_norm2 = [coef2*b for b in balans2]
    return balans_norm, balans_norm2


node_balans, edge_balans = get_balans(PATH_GRAPHS_JSONS, [0, 1, 2, 3, 4], [0, 1], "true_nodes", "true_edges")
print("BALANS NODE:", node_balans)
print("BALANS EDGE:", edge_balans[1]/edge_balans[0])