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
    "node_featch": 38,
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
        keywords = {
            # Класс 0: Текст / Text
            "text": 0, "основнойтекст": 0, "paragraph": 0, "content": 0,
            "body": 0, "описание": 0, "описательныйблок": 0, "описательныйтекст": 0,
            "описательнаячасть": 0, "description": 0, "простойтекст": 0,
            "текстовыйблок": 0, "контент": 0, "абзац": 0, "narrative": 0,
            "txt": 0, "оснтекст": 0, "para": 0, "par": 0, "cont": 0,
            "опис": 0, "desc": 0, "prose": 0, "nar": 0, "narr": 0,

            # Класс 1: Заголовок / Heading
            "title": 1, "header": 1, "заголовок": 1, "heading": 1,
            "subtitle": 1, "подзаголовок": 1, "section": 1, "раздел": 1,
            "chapter": 1, "глава": 1, "subheader": 1, "название": 1,
            "заголовок раздела": 1, "topic": 1, "h1": 1, "h2": 1,
            "h3": 1, "h4": 1, "h5": 1, "h6": 1, "hdr": 1, "head": 1,
            "titl": 1, "загл": 1, "подзагл": 1, "subhdr": 1, "sect": 1,
            "гл": 1, "разд": 1, "subttl": 1, "hdg": 1, "ttl": 1, "subhd": 1,

            # Класс 2: Список / List
            "list": 2, "список": 2, "bullet points": 2, "маркированныйсписок": 2,
            "numberedlist": 2, "нумерованныйсписок": 2, "items": 2,
            "элементысписка": 2, "перечисление": 2, "enumeration": 2,
            "checklist": 2, "check boxes": 2, "unorderedlist": 2,
            "orderedlist": 2, "перечень": 2, "пункты": 2, "lst": 2, "ul": 2,
            "ol": 2, "bul": 2, "numlst": 2, "марксписок": 2, "enum": 2,
            "chklst": 2, "пункт": 2, "item": 2, "elements": 2, "точки": 2,
            "bull": 2, "спис": 2, "lis": 2, "chkbox": 2,

            # Класс 3: Таблица / Table
            "table": 3, "таблица": 3, "spreadsheet": 3, "datatable": 3,
            "grid": 3, "матрица": 3, "сетка": 3, "tabular data": 3,
            "excel-like": 3, "pivot table": 3, "columns": 3, "столбцы": 3,
            "rows": 3, "строки": 3, "ячейки": 3, "cells": 3, "tbl": 3,
            "табл": 3, "col": 3, "row": 3, "cell": 3, "datatbl": 3,
            "pivot": 3, "столб": 3, "строка": 3, "colrow": 3, "excel": 3,
            "matrix": 3, "таблданные": 3, "colhd": 3, "tblstruct": 3,

            # Класс 4: Изображение / Figure
            "figure": 4, "fig": 4, "image": 4, "изображение": 4,
            "picture": 4, "рисунок": 4, "photo": 4, "фото": 4,
            "illustration": 4, "иллюстрация": 4, "графика": 4,
            "graphic": 4, "diagram": 4, "диаграмма": 4, "chart": 4,
            "график": 4, "screenshot": 4, "скриншот": 4, "visual": 4,
            "drawing": 4, "чертеж": 4, "schema": 4, "схема": 4,
            "img": 4, "pic": 4, "рис": 4, "илл": 4, "fgr": 4,
            "diagr": 4, "graph": 4, "скрин": 4, "scrnsht": 4,
            "схм": 4, "draw": 4, "figs": 4, "imgs": 4,
            "photos": 4, "граф": 4, "vis": 4, "thumb": 4, "preview": 4
        }
        new_nodes_feature = []
        for node, word in zip(model.nodes_feature, model.words):
            content = word.content.strip().lower().rstrip('.,;!?')
            is_keyword = content in keywords
            key_word_mark = keywords[content] if is_keyword else -1
            new_node = np.concatenate([node, [key_word_mark]])
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


