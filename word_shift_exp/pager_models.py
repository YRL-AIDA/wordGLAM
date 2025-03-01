"""
обязательно нужно реализовать
img2words_and_styles
words_and_styles2graph
get_img2phis
"""
import re

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
    "node_featch": 40,
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
        def topOrBottom(i, j):
            current_seg = model.words[i].segment
            neig_seg = model.words[j].segment
            top = current_seg.y_top_left < neig_seg.y_bottom_right
            bottom = current_seg.y_bottom_right > neig_seg.y_top_left
            return top or bottom

        A = model.A
        num_nodes = len(model.words)
        neig = [[] for _ in range(num_nodes)]

        for a1, a2 in zip(A[0], A[1]):
            if a1 < num_nodes and a2 < num_nodes:
                neig[a1].append(a2)
                neig[a2].append(a1)


        new_nodes_feature = []
        for i, node in enumerate(model.nodes_feature):
            segs = [model.words[j].segment for j in neig[i] if topOrBottom(i, j)]
            c = [s.get_center()[0] for s in segs]
            l = [s.x_top_left for s in segs]
            r = [s.x_bottom_right for s in segs]
            vec = [np.var(c) if len(c) > 1 else 0,
                   np.var(l) if len(l) > 1 else 0,
                   np.var(r) if len(r) > 1 else 0,]
            new_node = np.concatenate([node, vec])
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

class MyGLAMConverter(WordsAndStylesToGLAMBlocks):
    def __init__(self, conf):
        super().__init__(conf)
        self.my_extractor = MyExtractor()
        self.my_converter = MyConverter(self.graph_converter)

    def convert(self, input_model: WordsAndStylesModel, output_model: PhisicalModel) -> None:
        words = input_model.words   
        self.my_converter.convert(input_model, self.spgraph)
        self.my_extractor.extract(self.spgraph) # <--- Добавленый экстрактор
        graph = self.spgraph.to_dict()
        self.set_output_block(output_model, words, graph)

def get_img2phis(conf):
    unit_phis = PageModelUnit(id="phisical_model",
                              sub_model=PhisicalModel(),
                              extractors=[],
                              converters={"words_and_styles": MyGLAMConverter(conf=conf)})
    return PageModel(page_units=[
        unit_image,
        unit_words_and_styles,
        unit_phis
    ])


