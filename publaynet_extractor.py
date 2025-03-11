from config import extract, PATH_GRAPHS_JSONS, PATH_PUBLAYNET, PATH_PDF
import os 

extract(PATH_GRAPHS_JSONS, os.path.join(PATH_PUBLAYNET, 'train'), os.path.join(PATH_PDF, 'train'))
