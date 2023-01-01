
ann = {}
import json,os,sys
import argparse

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor
import jsonlines

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

model = 'damo/ofa_visual-question-answering_pretrain_huge_en'
preprocessor = OfaPreprocessor(model_dir=model)
ofa_pipe = pipeline(
    Tasks.visual_question_answering,
    model=model,
    preprocessor=preprocessor)

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default='', type=str,
                    help="prompt")
parser.add_argument("--device", default='', type=str,
                    help="device gpu")
parser.add_argument("--lang", default='ta', type=str,
                    help="lang")

args, _ = parser.parse_known_args()

lang =  args.lang
json_root = '/home/taoli1/marvl-code-forked/data/'+ lang + '/annotations_machine-translate/marvl-' + lang +'_gmt.jsonl'
print("json_root:", json_root)
if len(args.device) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args

# +
ann = load_annotations()
print("ann:", len(ann))

prompt = args.prompt.replace('#',' ').replace('*','\'')
# -

res = {}
i = 0
from tqdm import tqdm
for i in tqdm(range(len(ann))):
    # if i > 0:
    #     break
    v = ann[i]
    img_key = v['image_id_0'] + '##' + v['image_id_1']
    img_key = v['image_id_0'] + '##' + v['image_id_1']    
    img = "images/" + lang + '/' + img_key + '.jpg'
   
    text = prompt + v['sentence']
    if i == 0:
        print("img:", img)
        print("text:", text)
    input = {'image': img, 'text': text}
    result = ofa_pipe(input)
    reason = result[OutputKeys.TEXT][0]
    res[img_key] = reason
with open('./answers/' + lang + '-' + prompt + '_onestep.json', 'w') as f:
    json.dump( res, f)


