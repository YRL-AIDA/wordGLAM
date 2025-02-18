# from example_extract_exp import pager_models as exp_models
import os
import importlib
from dotenv import load_dotenv
dotenv_path = os.path.join("example_exp", '.env')

if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

EXPERIMENT = os.environ["EXPERIMENT"]
exp_models = importlib.import_module(f"{EXPERIMENT}.pager_models")

PATH_PUBLAYNET = "" #os.environ["PATH_PUBLAYNET"]

START = 0
FINAL = 10000

PATH_WORDS_AND_STYLES_JSONS = os.environ["PATH_WORDS_AND_STYLES_JSONS"]
PATH_GRAPHS_JSONS = os.environ["PATH_GRAPHS_JSONS"]

PATH_TEST_DATASET = os.environ["PATH_TEST_DATASET"]
PATH_TEST_IMAGES = PATH_TEST_DATASET #os.path.join(PATH_TEST_DATASET, "val")
PATH_TEST_JSON = os.path.join(PATH_TEST_DATASET, "result.json")

GLAM_NODE_MODEL = os.environ["GLAM_NODE_MODEL"]
GLAM_EDGE_MODEL = os.environ["GLAM_EDGE_MODEL"]
LOG_FILE = os.environ["LOG_FILE"]


PARAMS = importlib.import_module(f"{EXPERIMENT}.experiment_params").EXPERIMENT_PARAMS
SAVE_FREQUENCY = 40
SEG_K = 0.5

# try:
#     exp_param = importlib.import_module(f"{EXPERIMENT}.experiment_params")
#     PARAMS.update(exp_param.EXPERIMENT_PARAMS)
# except ImportError:
#     pass

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
