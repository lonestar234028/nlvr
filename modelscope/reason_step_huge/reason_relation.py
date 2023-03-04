# img_root = '/home/taoli1/nlvr/modelscope/images/'
img_root = '/vc_data/users/taoli1/mm/nlvr/'

json_root = '/vc_data/users/taoli1/mm/finetune/nlvr_test.json'
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
parser.add_argument("--prompt_post", default="", type=str,
                    help="prompt")
parser.add_argument("--device", default='', type=str,
                    help="device gpu")

args, _ = parser.parse_known_args()

if len(args.device) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args

with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))
# def get_img_path(v):
#     images = v['images']
#     k1 = images[0][len('test1/') :]
#     k2 = images[1][len('test1/') :]
#     k12 = k1 + "-"+ k2
#     img = img_root + k1 + "-"+ k2
#     return img

intermidiate_res = {}
with open('/home/taoli1/nlvr/modelscope/kpsall.json', 'r') as f:
    intermidiate_res = json.load(f)
print(len(intermidiate_res))

prompts =  []
with open('prompt.txt', 'r') as f:
    prompts = f.readlines()

print(len(intermidiate_res))



res = {}
prompt = args.prompt.replace('#',' ').replace('*','\'')
prompt_post = args.prompt_post.replace('#',' ').replace('*','\'')
magic = 699
from tqdm import tqdm
for i in tqdm(range(len(ann))):
    if i > magic:
        break
    if i < magic:
        continue
    v = ann[i]
    images = v['images']
    kw = intermidiate_res[v['sentence']]
    img_key = v['sentence'] + '##' + '##'.join(images)
    i += 1
    for prompt in prompts:
        x = prompt.split('\t')
        text = prompt
        if len(x) == 2:
            text = x[0]
        if len(kw) > 0:
            text = x[0] + kw[0] + x[1]
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
with open('reasons_v4/' + prompt + '.json', 'w') as f:
    json.dump(res, f)
