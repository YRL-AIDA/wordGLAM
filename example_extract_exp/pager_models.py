"""
обязательно нужно реализовать 
img2words_and_styles
words_and_styles2graph
get_img2phis
"""

from pager import PageModel, PageModelUnit
from pager.page_model.sub_models import BaseConverter, BaseExtractor, BaseSubModel
from pager.page_model.sub_models import ImageModel, WordsAndStylesModel, SpGraph4NModel, BaseConverter
from pager.page_model.sub_models import ImageToWordsAndCNNStyles,  WordsAndStylesToSpGraph4N
from pager.page_model.sub_models import PhisicalModel, WordsAndStylesToGLAMBlocks
from pager import PageModel, PageModelUnit, WordsAndStylesModel, SpGraph4NModel, WordsAndStylesToSpGraph4N, WordsAndStylesToSpDelaunayGraph
import os 

PATH_STYLE_MODEL = os.getenv("PATH_STYLE_MODEL", "default_path")
WITH_TEXT = True
TYPE_GRAPH = "4N"

class MyConverter(BaseConverter):
    def __init__(self, old_converter):
        self.old_converter = old_converter

    def convert(self, input_model: BaseSubModel, output_model: BaseSubModel)-> None:
        self.old_converter.convert(input_model, output_model)
        output_model.words = input_model.words # Информация о словах не передается в SpGraph4NModel
        
class MyExtractor(BaseExtractor):
    def extract(self, model: BaseSubModel) -> None:
#                                      v-- Информция здесь уже есть, так как мы переделали конвертер
        for node, word in zip(model.nodes_feature, model.words):
            node[-1] = 1 if word.content[0].isupper() else 0 # Меняю последний признак на такой

        

unit_image = PageModelUnit(id="image", 
                               sub_model=ImageModel(), 
                               converters={}, 
                               extractors=[])
    
conf_words_and_styles = {"path_model": PATH_STYLE_MODEL,"lang": "eng+rus", "psm": 4, "oem": 3, "k": 4 }
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
                            extractors=[MyExtractor()],  #<--- Новый экстрактор
                            converters={"words_and_styles": MyConverter(WordsAndStylesToSpDelaunayGraph(conf_graph) #<---Изменения в конвертере
                                                            if TYPE_GRAPH == "Delaunay" else 
                                                            WordsAndStylesToSpGraph4N(conf_graph)) })
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


