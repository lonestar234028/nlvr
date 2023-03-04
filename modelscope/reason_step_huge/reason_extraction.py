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
parser.add_argument("--prompt", default=2, type=str,
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

res = {}
prompt = args.prompt.replace('#',' ').replace('*','\'')
from tqdm import tqdm
for i in tqdm(range(len(ann))):
    if i > 0:
        break
    v = ann[i]
    images = v['images']
    img_key = v['sentence'] + '##' + '##'.join(images)
    i += 1
    text = prompt
    reason = text + '##'
    input = {'image': img_root + images[0], 'text': text}
    print("input1:",input)
    result = ofa_pipe(input)
    reason += result[OutputKeys.TEXT][0] + '##'
    input = {'image': img_root + images[1], 'text': text}
    print("input2:",input)
    result = ofa_pipe(input)
    reason += result[OutputKeys.TEXT][0]
    res[img_key] = reason
with open('reasons/' + prompt + '.json', 'w') as f:
    json.dump(res, f)
