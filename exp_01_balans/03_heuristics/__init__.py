import sys, os
sys.path.append("..")
from exp_00_base import *

PUBLAYNET_IMBALANCE = [
    3.00, #figure  - не много слов
    0.01, #text - самый большой представитель
    0.1,  #title - заголовки частые, но меньше текста раз в 10
    1.00, #list - списки чуть больше заголовков, но редкие
    0.89   #table - таблицы редкие, но чаще списков
]

EXPERIMENT_PARAMS["loss_params"]["publaynet_imbalance"] =PUBLAYNET_IMBALANCE
EXPERIMENT_PARAMS["loss_params"]["edge_imbalance"] = 0.2