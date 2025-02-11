"""
обязательно нужно реализовать 
img2words_and_styles
words_and_styles2graph
get_img2phis
"""

from pager import PageModel, PageModelUnit
from pager.page_model.sub_models import ImageModel, WordsAndStylesModel, SpGraph4NModel
from pager.page_model.sub_models import ImageToWordsAndCNNStyles,  WordsAndStylesToSpGraph4N
from pager.page_model.sub_models import PhisicalModel, WordsAndStylesToGLAMBlocks
from pager import PageModel, PageModelUnit, WordsAndStylesModel, SpGraph4NModel, WordsAndStylesToSpGraph4N, WordsAndStylesToSpDelaunayGraph
import os 

PATH_STYLE_MODEL = os.environ["PATH_STYLE_MODEL"]
WITH_TEXT = True
TYPE_GRAPH = "4N"

unit_image = PageModelUnit(id="image", 
                               sub_model=ImageModel(), 
                               converters={}, 
                               extractors=[])
    
conf_words_and_styles = {"path_model": PATH_STYLE_MODEL,"lang": "eng+rus", "psm": 4, "oem": 3, "k": 4 }
unit_words_and_styles = PageModelUnit(id="word_and_style", 
                            sub_model=WordsAndStylesModel(), 
                            converters={"image": ImageToWordsAndCNNStyles(conf_words_and_styles)}, 
                            extractors=[]),

conf_graph = {"with_text": True} if WITH_TEXT else None
unit_graph = PageModelUnit(id="graph", 
                            sub_model=SpGraph4NModel(), 
                            extractors=[],  
                            converters={"words_and_styles": WordsAndStylesToSpDelaunayGraph(conf_graph) 
                                                            if TYPE_GRAPH == "Delaunay" else 
                                                            WordsAndStylesToSpGraph4N(conf_graph) }),

img2words_and_styles = PageModel([
    unit_image, 
    unit_words_and_styles
])

words_and_styles2graph = PageModel([
    unit_words_and_styles,
    unit_graph
])


def get_img2phis(conf):
    unit_phis = PageModelUnit(id="phisical_model", 
                        sub_model=PhisicalModel(), 
                        extractors=[], 
                        converters={"words_and_styles_model": WordsAndStylesToGLAMBlocks(conf=conf)})
    return PageModel([
        unit_image, 
        unit_words_and_styles,
        unit_phis
    ])


