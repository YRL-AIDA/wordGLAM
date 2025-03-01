"""
обязательно нужно реализовать
img2words_and_styles
words_and_styles2graph
get_img2phis
"""
import numpy as np
from pager import PageModel, PageModelUnit
from pager.page_model.sub_models import BaseConverter, BaseExtractor, BaseSubModel
from pager.page_model.sub_models import ImageModel, WordsAndStylesModel, SpGraph4NModel, BaseConverter
from pager.page_model.sub_models import ImageToWordsAndCNNStyles, WordsAndStylesToSpGraph4N
from pager.page_model.sub_models import PhisicalModel, WordsAndStylesToGLAMBlocks
from pager import PageModel, PageModelUnit, WordsAndStylesModel, SpGraph4NModel, WordsAndStylesToSpGraph4N, \
    WordsAndStylesToSpDelaunayGraph
import os
from dotenv import load_dotenv

load_dotenv(override=True)

PATH_STYLE_MODEL = os.environ["PATH_STYLE_MODEL"]

WITH_TEXT = True
TYPE_GRAPH = "4N"
EXPERIMENT_PARAMS = {
    "node_featch": 41,
    "edge_featch": 2,
    "epochs": 5,
    "batch_size": 5,
    "learning_rate": 0.05,
    "H1": [256, 128, 64, 32, 16, 8],
    "H2": [8],
    "seg_k": 0.5
}


class MyConverter(BaseConverter):
    def __init__(self, old_converter):
        self.old_converter = old_converter

    def convert(self, input_model: BaseSubModel, output_model: BaseSubModel) -> None:
        self.old_converter.convert(input_model, output_model)
        output_model.words = input_model.words  # Информация о словах не передается в SpGraph4NModel


class MyExtractor(BaseExtractor):
    def extract(self, model: BaseSubModel) -> None:
        new_nodes_feature = []
        for node, word in zip(model.nodes_feature, model.words):
            sign = []
            sign.append(1) if "." in word.content else sign.append(0)
            sign.append(1) if "," in word.content else sign.append(0)
            sign.append(1) if ";" in word.content else sign.append(0)
            sign.append(1) if ":" in word.content else sign.append(0)
            new_node = np.concatenate([node, sign])
            new_nodes_feature.append(new_node)
        new_nodes_feature = np.array(new_nodes_feature)
        model.nodes_feature = new_nodes_feature



unit_image = PageModelUnit(id="image",
                           sub_model=ImageModel(),
                           converters={},
                           extractors=[])

conf_words_and_styles = {"path_model": PATH_STYLE_MODEL, "lang": "eng+rus", "psm": 4, "oem": 3, "k": 4}
unit_words_and_styles = PageModelUnit(id="words_and_styles",
                                      sub_model=WordsAndStylesModel(),
                                      converters={"image": ImageToWordsAndCNNStyles(conf_words_and_styles)},
                                      extractors=[])
unit_words_and_styles_start = PageModelUnit(id="words_and_styles",
                                            sub_model=WordsAndStylesModel(),
                                            converters={},
                                            extractors=[])
conf_graph = {"with_text": True} if WITH_TEXT else None
unit_graph = PageModelUnit(id="graph",
                           sub_model=SpGraph4NModel(),
                           extractors=[MyExtractor()],  # <--- Новый экстрактор
                           converters={"words_and_styles": MyConverter(
                               WordsAndStylesToSpDelaunayGraph(conf_graph)  # <---Изменения в конвертере
                               if TYPE_GRAPH == "Delaunay" else
                               WordsAndStylesToSpGraph4N(conf_graph))})
img2words_and_styles = PageModel(page_units=[
    unit_image,
    unit_words_and_styles
])

words_and_styles2graph = PageModel(page_units=[
    unit_words_and_styles_start,
    unit_graph
])


def get_img2phis(conf):
    unit_phis = PageModelUnit(id="phisical_model",
                              sub_model=PhisicalModel(),
                              extractors=[],
                              converters={"words_and_styles": WordsAndStylesToGLAMBlocks(conf=conf)})
    return PageModel(page_units=[
        unit_image,
        unit_words_and_styles,
        unit_phis
    ])


