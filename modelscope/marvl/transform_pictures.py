img_root = '/vc_data/users/taoli1/topic/marvl-images/zh/images/'
json_root = '/home/taoli1/marvl-code-forked/data/zh/annotations_machine-translate/marvl-zh_gmt.jsonl'

ann = {}
import json
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import jsonlines
import os
# ImageFile.LOAD_TRUNCATED_IMAGES = True

resolution = 480
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std),
    ])

    
def load_annotations():
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

def load_images_path():
    paths = {}
    for dirs in os.listdir(img_root):
        for dir in os.listdir(img_root + dirs):
            path_dir = (img_root + dirs + '/' + dir)
            paths[dir.split(".")[0]] = path_dir
    return paths



def merge_picture(pic1, pic2):
    patch_img1 = patch_resize_transform(pic1)
    patch_img2 = patch_resize_transform(pic2)
    patch_img = torch.cat((patch_img1, patch_img2), dim=-1)
    # return patch_img.unsqueeze(0), patch_img1.unsqueeze(0), patch_img2.unsqueeze(0)
    return patch_img

def insert_image(item, paths):
    pic1 = Image.open(paths[item['image_id_0']])
    pic2 = Image.open(paths[item['image_id_1']])
    return merge_picture(pic1, pic2)
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES=True

ann = load_annotations()
imgid2paths = load_images_path()
print("imgid2paths:", len(imgid2paths))
print("ann:",len(ann))

from tqdm import tqdm
for i in tqdm(range(len(ann))):
    # if i > 0:
    #     break
    v = ann[i]
    img_key = v['image_id_0'] + '##' + v['image_id_1']
    patch_img = insert_image(v, imgid2paths)
    
    save_image(patch_img, "images/zh/" + img_key + '.jpg')
    i += 1