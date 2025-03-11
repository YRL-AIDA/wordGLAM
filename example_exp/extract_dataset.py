import os 
from pager.page_model.sub_models import JsonWithFeatchs
from pager.page_model.sub_models.dtype import ImageSegment
from .pager_models import words_and_styles2graph, img2words_and_styles,ws2g_converter
import numpy as np



def featch_words_and_styles(img_name):
    img2words_and_styles.read_from_file(img_name)
    img2words_and_styles.extract()
    return [
        img2words_and_styles.page_units[-1].sub_model.to_dict(is_vec=True)['styles'],
        img2words_and_styles.page_units[-1].sub_model.to_dict(is_vec=True)['words']
    ]

def featch_A(path_json):
    words_and_styles2graph.read_from_file(path_json)
    words_and_styles2graph.extract()
    graph_json = words_and_styles2graph.to_dict()
    return [
        graph_json['A']
    ]

def nodes_feature(styles, words):
    fonts = dict()
    for st in styles:
        fonts[st['id']] = st['font2vec']
    style_vec = np.array([fonts[w['style_id']] for w in words])
    text_vec = ws2g_converter.word2vec([w['content'] for w in words])

    nodes_feature = np.concat([style_vec, text_vec], axis=1)
    return [nodes_feature.tolist()]

def edges_feature(A, words):
    edges_featch = []
    for i, j in zip(A[0], A[1]):
        w1 = ImageSegment(dict_2p= words[i]['segment'])
        w2 = ImageSegment(dict_2p= words[j]['segment'])
        edges_featch.append([w1.get_angle_center(w2), w1.get_min_dist(w2)])
    # print(edges_featch)
    return [edges_featch]
      

def true_class_from_publaynet(blocks, words, A):
    # Вспомогательные функции
    def get_block_seg(json_block):
        seg = ImageSegment(dict_p_size=json_block)
        seg.add_info("label", json_block["label"])
        return seg
    def get_word_seg(json_word):
        return ImageSegment(dict_2p=json_word['segment'])
    
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
    
    # Получение верных меток из датасета PubLayNet
    block_segs = [get_block_seg(bl) for bl in blocks]
    word_segs = [get_word_seg(w) for w in words]
    edges_ind = [is_one_block(word_segs[i],word_segs[j], block_segs) for i, j in zip(A[0], A[1])]
    nodes_ind = [get_class_node(w, block_segs) for w in word_segs]
    return [edges_ind, nodes_ind]

def extract(path_dataset, path_img_publaynet=None, path_pdf_publaynet=None):
    json_with_featchs = JsonWithFeatchs()
    files = os.listdir(path_dataset)
    N = len(files)
    for i, file in enumerate(files):
        print(file, end=' ')
        try:
            path_json = os.path.join(path_dataset, file)
            json_with_featchs.read_from_file(path_json)
            path_img = os.path.join(path_img_publaynet, json_with_featchs.json['file_name'])
            
            json_with_featchs.add_featchs(lambda: featch_words_and_styles(path_img), names=['styles', 'words'], 
                                is_reupdate=False, rewrite=True)
            
            json_with_featchs.add_featchs(lambda: featch_A(path_json), names=['A'], 
                                is_reupdate=False, rewrite=True)
            
            json_with_featchs.add_featchs(lambda: nodes_feature(json_with_featchs.json['styles'], json_with_featchs.json['words']), names=['nodes_feature'], 
                                is_reupdate=False, rewrite=True) 
            
            json_with_featchs.add_featchs(lambda: edges_feature(json_with_featchs.json['A'], json_with_featchs.json['words']), names=['edges_feature'], 
                                is_reupdate=False, rewrite=True) 
            
            json_with_featchs.add_featchs(lambda: true_class_from_publaynet(json_with_featchs.json['blocks'], json_with_featchs.json['words'], json_with_featchs.json['A']), names=["true_edges", "true_nodes"], 
                                is_reupdate=False, rewrite=True)
            
        except:
            print("error in ", file)
        
        print(f" ({(i+1)/N*100:.2f} %) "+20*" ", end='\n')  