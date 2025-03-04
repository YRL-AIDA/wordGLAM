from publaynet_reader import PubLayNetDataset 
from pager.page_model.sub_models.dtype import ImageSegment
from config import get_preprocessing_models, PATH_PUBLAYNET, START, FINAL, PATH_WORDS_AND_STYLES_JSONS, PATH_GRAPHS_JSONS, PATH_PDF
import os
import json
LABELS = {
    'figure': 0,
    'text': 1,
    'title': 2,
    'list': 3,
    'table': 4
}
if __name__ == "__main__":   
    pdf2words_and_styles, words_and_styles2graph = get_preprocessing_models() # На самом деле первое pdf2words_and_styles
    # Первый шаг, получаем информацию из картинок и разметки PubLayNet
    def get_words_and_styles(img_path):
        
        try:
            name_pdf = ".".join(os.path.basename(img_path).split('.')[:-1])+'.pdf'
            pdf_path = os.path.join(PATH_PDF, name_pdf)
            pdf2words_and_styles.read_from_file(pdf_path)
            pdf2words_and_styles.extract()
            return pdf2words_and_styles.page_units[-1].sub_model.to_dict(is_vec=True)
        except:
            return {}

    if not os.path.exists(PATH_WORDS_AND_STYLES_JSONS):    
        with open(os.path.join(PATH_PUBLAYNET, "train.json"), "r") as f:
            dataset = json.load(f)
        
        # Индексы изображений по id
        id2index_image = dict()
        for i, im in enumerate(dataset["images"]):
            id2index_image[im["id"]] = i
            im["blocks"] = []
        
        # Имена категорий по id
        id2name_category = dict()
        for c in dataset["categories"]:
            id2name_category[c["id"]] = c["name"]
        print(id2name_category)
        del dataset["categories"]

        # Аннотации для каждого изображения
        for an in dataset["annotations"]:
            block = dict()
            block["label"] = LABELS[id2name_category[an["category_id"]]]
            block["x_top_left"] = int(an["bbox"][0])
            block["y_top_left"] = int(an["bbox"][1])
            block["width"] = int(an["bbox"][2])
            block["height"] = int(an["bbox"][3])
            dataset["images"][id2index_image[an["image_id"]]]["blocks"].append(block)
            del an
        
        # Обработка изображения
        N = len(dataset["images"])
        os.mkdir(PATH_WORDS_AND_STYLES_JSONS)
        os.mkdir(os.path.join(PATH_WORDS_AND_STYLES_JSONS, "train"))
        for k, im in enumerate(dataset["images"]):
            name = im["file_name"]
            img_path = os.path.join(PATH_PUBLAYNET, "train", name)
            words_and_styles = get_words_and_styles(img_path)
            if words_and_styles == {}:
                continue
            im["additional_info"] = words_and_styles
            with open(os.path.join(PATH_WORDS_AND_STYLES_JSONS, "train", name+'.json'), "w") as f:
                json.dump(im, f)
            print(f"{(k+1)/N*100:.2f}%"+" "*10, end="\r")
        print("\n")

    
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
    if not os.path.exists(PATH_GRAPHS_JSONS):
        os.mkdir(PATH_GRAPHS_JSONS)
        dataset_dir = os.path.join(PATH_WORDS_AND_STYLES_JSONS, 'train')
        files = os.listdir(dataset_dir)
        N = len(files)
        for i, json_file in enumerate(files):
            try:
                graph = get_graph_from_file(os.path.join(dataset_dir, json_file), words_and_styles2graph)
                path_graph = os.path.join(PATH_GRAPHS_JSONS, json_file)
                with open(path_graph, "w") as f:
                    json.dump(graph, f)
            except:
                print("error in ", json_file)
            print(f"{(i+1)/N*100:.2f} %"+20*" ", end='\r')     
    
    
