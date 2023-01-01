
import jsonlines
import os

def load_annotations(lang):
    json_root = '/home/taoli1/marvl-code-forked/data/'+ lang + '/annotations_machine-translate/marvl-' + lang +'_gmt.jsonl'

    items = []
    with jsonlines.open(json_root) as reader:
        # Build an index which maps image id with a list of hypothesis annotations.
        count = 0
        for annotation in reader:
            dictionary = {}
            dictionary["image_id_0"] = annotation["left_img"].split("/")[-1].split(".")[0]
            dictionary["image_id_1"] = annotation["right_img"].split("/")[-1].split(".")[0]
            dictionary["question_id"] = count

            dictionary["sentence"] = str(annotation["caption"])
            dictionary["labels"] = [int(annotation["label"])]
            dictionary["concept"] = str(annotation["concept"])
            dictionary["scores"] = [1.0]
            dictionary["ud"] = str(annotation["id"])
            items.append(dictionary)
            count += 1
            if count < 2:
                print("loading_annotations: ")
                print(dictionary)
    return items



def load_images_path(img_root):
    paths = {}
    for dirs in os.listdir(img_root):
        if dirs.startswith('.DS'):
            continue
        for dir in os.listdir(img_root + dirs):
            path_dir = (img_root + dirs + '/' + dir)
            paths[dir.split(".")[0]] = path_dir
    return paths
