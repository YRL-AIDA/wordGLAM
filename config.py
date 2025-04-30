# from example_extract_exp import pager_models as exp_models
import os
import importlib
from dotenv import load_dotenv
# dotenv_path = os.path.join('.env')

# if os.path.exists(dotenv_path):
#     load_dotenv(dotenv_path)
load_dotenv(override=True)

EXPERIMENT = os.environ["EXPERIMENT"]
exp = importlib.import_module(f"{'.'.join(EXPERIMENT.split('/'))}")

# print(f"{EXPERIMENT}.pager_models")
# exp_models = importlib.import_module(f"{EXPERIMENT}.pager_models")
# exp_dataset = importlib.import_module(f"{EXPERIMENT}.extract_dataset")
# exp_torchmodel = importlib.import_module(f"{EXPERIMENT}.torch_model")
# TorchModel = exp_torchmodel.TorchModel
# CustomLoss = exp_torchmodel.CustomLoss
TorchModel = exp.TorchModel
CustomLoss = exp.CustomLoss

PATH_PUBLAYNET = os.environ["PATH_PUBLAYNET"]
PATH_PDF = os.environ["PATH_PDF"]

START = os.environ["START"]
FINAL = os.environ["FINAL"]

PATH_WORDS_AND_STYLES_JSONS = os.environ["PATH_WORDS_AND_STYLES_JSONS"]
PATH_GRAPHS_JSONS = os.environ["PATH_GRAPHS_JSONS"]

PATH_TEST_DATASET = os.environ["PATH_TEST_DATASET"]
PATH_TEST_IMAGES = os.environ["PATH_TEST_IMAGES"]
PATH_TEST_JSON = os.environ["PATH_TEST_JSON"]
PATH_TEST_PDF = os.environ["PATH_TEST_PDF"]

GLAM_NODE_MODEL = os.path.join(EXPERIMENT, os.environ["GLAM_NODE_MODEL"])
GLAM_EDGE_MODEL = os.path.join(EXPERIMENT, os.environ["GLAM_EDGE_MODEL"])
GLAM_MODEL = os.path.join(EXPERIMENT, os.environ["GLAM_MODEL"])
LOG_FILE = os.path.join(EXPERIMENT, "log.txt")


# PARAMS = importlib.import_module(f"{EXPERIMENT}.pager_models").EXPERIMENT_PARAMS
PARAMS = exp.EXPERIMENT_PARAMS
SAVE_FREQUENCY = int(os.environ["SAVE_FREQUENCY"])



PUBLAYNET_IMBALANCE = exp.PUBLAYNET_IMBALANCE
EDGE_IMBALANCE = exp.EDGE_IMBALANCE 
EDGE_COEF = exp.EDGE_COEF 
NODE_COEF = exp.NODE_COEF


LOSS_PARAMS = {
    "publaynet_imbalance": PUBLAYNET_IMBALANCE,
    "edge_imbalance": EDGE_IMBALANCE,
    "edge_coef": EDGE_COEF,
    "node_coef": NODE_COEF,
}
PARAMS["loss_params"] = LOSS_PARAMS


extract = exp.extract

def get_preprocessing_models():
    return exp.img2words_and_styles, exp.words_and_styles2graph

def get_final_model():
    conf = PARAMS
    conf['path_model'] = GLAM_MODEL
    return exp.get_img2phis(conf)
