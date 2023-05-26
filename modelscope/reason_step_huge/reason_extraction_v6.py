# img_root = '/home/taoli1/nlvr/modelscope/images/'
import data_prepare as nlvr_data

img_root = str(nlvr_data.d.img_root_abs)
json_root = nlvr_data.d.test_json_abs_path
print("img_root->", img_root)
print("json_root->", json_root)
ann = {}
import json,os,sys
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor
import argparse

model = 'damo/ofa_visual-question-answering_pretrain_huge_en'
preprocessor = OfaPreprocessor(model_dir=model)
ofa_pipe = pipeline(
    Tasks.visual_question_answering,
    model=model,
    preprocessor=preprocessor)

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="what's the scenario about ", type=str,
                    help="prompt")
parser.add_argument("--partpre", default=0, type=str,
                    help="partpre")
parser.add_argument("--partpost", default=6967, type=str,
                    help="partpost")
parser.add_argument("--prompt_post", default="", type=str,
                    help="prompt")
parser.add_argument("--device", default='', type=str,
                    help="device gpu")

args, _ = parser.parse_known_args()
print(args)
if len(args.device) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args

with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))


prompts =  []
with open('prompt_v2.txt', 'r') as f:
    prompts = f.readlines()


res = {}
magic = 6
print("prompts:", prompts)
start = args.partpre
end = args.partpost
if isinstance(start, str):
    start = int(start)
if isinstance(end, str):
    end = int(end)

from tqdm import tqdm
for prompt in prompts:
    for i in tqdm(range(start,end)):
       
        # if i > magic:
        #     break
        # if i < magic:
        #     continue
        v = ann[i]
        images = v['images']
       
        img_key = v['sentence'] + '##' + '##'.join(images)
        text = prompt.strip()
      
        reason = text + '##'
        input1 = {'image': img_root + images[0], 'text': text}
        result = ofa_pipe(input1)
        reason += result[OutputKeys.TEXT][0] + '##'
        input2 = {'image': img_root + images[1], 'text': text}
        
        result = ofa_pipe(input2)
        reason += result[OutputKeys.TEXT][0]
        res[img_key] = reason
        print("input1:",input1)
        print("input2:",input2)
        
    with open('reasons_v6/' + prompt + '_part_'+ str(start) +'_' +  str(end) + '.json', 'w') as f:
        json.dump(res, f)
