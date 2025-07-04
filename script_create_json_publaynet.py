from config import PATH_PUBLAYNET, PATH_GRAPHS_JSONS
import os
import json
LABELS = {
    'figure': 0,
    'text': 1,
    'title': 2,
    'list': 3,
    'table': 4
}
    
def create_jsons_publaynet():
    if  os.path.exists(PATH_GRAPHS_JSONS):  
        print(f"DIR EXISTS {PATH_GRAPHS_JSONS}")
        return  
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
    N = len(dataset["annotations"])
    for k, an in enumerate(dataset["annotations"]):
        block = dict()
        block["label"] = LABELS[id2name_category[an["category_id"]]]
        block["x_top_left"] = int(an["bbox"][0])
        block["y_top_left"] = int(an["bbox"][1])
        block["width"] = int(an["bbox"][2])
        block["height"] = int(an["bbox"][3])
        dataset["images"][id2index_image[an["image_id"]]]["blocks"].append(block)
        del an
        print(f"read annotation: {(k+1)/N*100:.2f}%"+" "*10, end="\r")
    print("\n")
    
        # Обработка изображения
    N = len(dataset["images"])
    os.mkdir(PATH_GRAPHS_JSONS)
    for k, im in enumerate(dataset["images"]):
        name = im["file_name"]
        with open(os.path.join(PATH_GRAPHS_JSONS, name+'.json'), "w") as f:
            json.dump(im, f)
        print(f"write json: {(k+1)/N*100:.2f}%"+" "*10, end="\r")
    print("\n")

if __name__ == "__main__":   
    # Первый шаг, получаем информацию из картинок и разметки PubLayNet
    print("create jsons")
    create_jsons_publaynet()
    
    
