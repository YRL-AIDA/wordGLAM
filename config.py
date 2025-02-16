# from example_extract_exp import pager_models as exp_models
import os
import importlib

# Выбор эксперимента (поменять на нужный)
EXPERIMENT = os.getenv("EXPERIMENT", "example_exp")
exp_models = importlib.import_module(f"{EXPERIMENT}.pager_models")

# Остальные переменные конфига
PATH_PUBLAYNET = os.getenv("PATH_PUBLAYNET", "")
START = 0
FINAL = 10000
PATH_WORDS_AND_STYLES_JSONS = os.getenv("PATH_WORDS_AND_STYLES_JSONS", "default_path")
PATH_GRAPHS_JSONS = os.getenv("PATH_GRAPH_JSONS", "graphs_jsons")

PATH_TEST_DATASET = os.getenv("PATH_PUBLAYNET", "")
PATH_TEST_IMAGES = PATH_TEST_DATASET #os.path.join(PATH_TEST_DATASET, "val")
PATH_TEST_JSON = os.path.join(PATH_TEST_DATASET, "result.json")

GLAM_NODE_MODEL = os.path.join(EXPERIMENT, "mini_data_glam_node_model")
GLAM_EDGE_MODEL = os.path.join(EXPERIMENT, "mini_data_glam_edge_model")
LOG_FILE = os.path.join(EXPERIMENT, "log.txt")
PARAMS = {
        "node_featch": 37,
        "edge_featch": 2,
        "epochs": 400,
        "batch_size": 500,
        "learning_rate": 0.05,
        "H1": [64, 64, 64, 64, 64],
        "H2": [64]
    }
SAVE_FREQUENCY = 40
SEG_K = 0.5

try:
    exp_param = importlib.import_module(f"{EXPERIMENT}.experiment_params")
    PARAMS.update(exp_param.EXPERIMENT_PARAMS)
except ImportError:
    pass

def get_preprocessing_models():
    return exp_models.img2words_and_styles, exp_models.words_and_styles2graph

def get_final_model():
    return exp_models.get_img2phis(conf={
            "path_node_gnn": GLAM_NODE_MODEL,
            "path_edge_linear": GLAM_EDGE_MODEL,
            "H1": PARAMS["H1"],
            "H2": PARAMS["H2"],
            "node_featch": PARAMS["node_featch"],
            "edge_featch": PARAMS["edge_featch"],
            "seg_k": SEG_K

    })
