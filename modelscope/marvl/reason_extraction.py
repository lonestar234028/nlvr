# img_root = '/home/taoli1/nlvr/modelscope/images/'
img_root = '/vc_data/users/taoli1/mm/nlvr/'

json_root = '/vc_data/users/taoli1/mm/finetune/nlvr_test.json'
ann = {}
import json,os,sys
import torch

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="Firstly,", type=str,
                    help="prompt")
parser.add_argument("--device", default='', type=str,
                    help="device gpu")


parser.add_argument("--lang", default='ta', type=str,
                    help="lang")

args, _ = parser.parse_known_args()

lang =  args.lang
if len(args.device) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args

from dataset import *
ann = load_annotations(args.lang)
print("ann:", len(ann))


from torchvision import transforms
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor
model = 'damo/ofa_visual-question-answering_pretrain_large_en'
preprocessor = OfaPreprocessor(model_dir=model)
ofa_pipe = pipeline(
    Tasks.visual_question_answering,
    model=model,
    preprocessor=preprocessor)

# def get_img_path(v):
#     images = v['images']
#     k1 = images[0][len('test1/') :]
#     k2 = images[1][len('test1/') :]
#     k12 = k1 + "-"+ k2
#     img = img_root + k1 + "-"+ k2
#     return img

res = {}
prompt = args.prompt.replace('#',' ').replace('*','\'')
img_root = '/vc_data/users/taoli1/topic/marvl-images/' + lang + '/images/'
imgid2paths = load_images_path(img_root)

from tqdm import tqdm
for i in tqdm(range(len(ann))):
    # if i > 0:
    #     break
    v = ann[i]
    img_key = v['image_id_0'] + '##' + v['image_id_1']
    text = prompt
    reason = text + '##'
    print("imgid2paths[v['image_id_0']]:", imgid2paths[v['image_id_0']])
    input = {'image': imgid2paths[v['image_id_0']], 'text': text}
    result = ofa_pipe(input)
    reason += result[OutputKeys.TEXT][0] + '##'
    input = {'image': imgid2paths[v['image_id_0']], 'text': text}
    result = ofa_pipe(input)
    reason += result[OutputKeys.TEXT][0]
    res[img_key] = reason
with open('reasons/' + lang +'_' + prompt + '.json', 'w') as f:
    json.dump(res, f)
