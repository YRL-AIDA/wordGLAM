from publaynet_reader import PubLayNetDataset 
from pager.page_model.sub_models.dtype import ImageSegment
from config import get_preprocessing_models, PATH_PUBLAYNET, START, FINAL, PATH_WORDS_AND_STYLES_JSONS, PATH_GRAPHS_JSONS
import os
import json


if __name__ == "__main__":   
    img2words_and_styles, words_and_styles2graph = get_preprocessing_models()
    # Первый шаг, получаем информацию из картинок и разметки PubLayNet
    def get_words_and_styles(img_path):
        img2words_and_styles.read_from_file(img_path)
        img2words_and_styles.extract()
        return img2words_and_styles.page_units[-1].sub_model.to_dict(is_vec=True)
    
    pln_ds = PubLayNetDataset(PATH_PUBLAYNET, PATH_WORDS_AND_STYLES_JSONS)
    pln_ds.create_tmp_annotation_jsons(path_tmp_dataset=PATH_WORDS_AND_STYLES_JSONS, 
                                       fun_additional_info=get_words_and_styles, 
                                       start_min_category= START, finish_min_category=FINAL)
    
    # Второй шаг, строим граф из информации от картинок

    def get_graph_from_file(file_name, page_model):
        # Вспомогательные функции
        def get_block_seg(json_block):
            seg = ImageSegment(dict_p_size=json_block)
            seg.add_info("label", json_block["label"])
            return seg

        def is_one_block(word1, word2, blocks):
            for block in blocks:
                if block.is_intersection(word1) and block.is_intersection(word2):
                    return 1
            return 0

        def get_class_node(word, blocks):
            for block in blocks:
                if block.is_intersection(word):
                    return block.get_info("label")
            return -1
        with open(file_name, "r") as f:
            info_img = json.load(f)
        
        # Получение информации с прошлого шага обработки img2words_and_styles
        pager_rez = info_img["additional_info"]
        page_model.from_dict(pager_rez)
        page_model.extract()
        graph = page_model.to_dict()

        # Получение верных меток из датасета PubLayNet
        publaynet_rez = info_img["blocks"]
        block_segs = [get_block_seg(bl) for bl in publaynet_rez]
        words = [w.segment for w in page_model.page_units[0].sub_model.words]
        edges_ind = [is_one_block(words[i],words[j], block_segs) for i, j in zip(graph["A"][0], graph["A"][1])]
        nodes_ind = [get_class_node(w, block_segs) for w in words]
        graph["true_edges"] = edges_ind
        graph["true_nodes"] = nodes_ind
        return graph

    files = os.listdir(PATH_WORDS_AND_STYLES_JSONS)
    N = len(files)
    for i, json_file in enumerate(files):
        try:
            graph = get_graph_from_file(os.path.join(PATH_WORDS_AND_STYLES_JSONS, json_file), words_and_styles2graph)
            path_graph = os.path.join(PATH_GRAPHS_JSONS, json_file)
            with open(path_graph, "w") as f:
                json.dump(graph, f)
        except:
            print("error in ", json_file)
        print(f"{(i+1)/N*100:.2f} %"+20*" ", end='\r')     
    
    