 # -*- coding: utf-8 -*-
img_root = '/home/taoli1/nlvr/modelscope/images/'
json_root = '/vc_data/users/taoli1/mm/finetune/nlvr_test.json'

ann = {}
import json,os,sys
import argparse

# +

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import os
import shutil
from modelscope.utils.hub import snapshot_download
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile

# -

pretrained_model = 'damo/ofa_pretrain_large_en'
pretrain_path = snapshot_download(pretrained_model, revision='v1.0.0')
# task_model = 'damo/ofa_image-caption_coco_large_en'
# task_path = snapshot_download(task_model)
# shutil.copy(os.path.join(task_path, ModelFile.CONFIGURATION), 
#             os.path.join(pretrain_path, ModelFile.CONFIGURATION))
ofa_pipe = pipeline(Tasks.image_captioning, model=pretrain_path)


parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default='', type=str,
                    help="prompt")
parser.add_argument("--device", default='', type=str,
                    help="device gpu")

args, _ = parser.parse_known_args()

if len(args.device) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args

# +
with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))

prompt = args.prompt.replace('#',' ').replace('*','\'')

# +
res = {}
i = 0
from tqdm import tqdm
for i in tqdm(range(len(ann))):
    if i > 0:
        break
    v = ann[i]
    images = v['images']
    img_key = v['sentence'] + '##' + '##'.join(images)
  
    i += 1
    k1 = images[0][len('test1/') :]
    k2 = images[1][len('test1/') :]
    img = img_root + k1 + "-"+ k2
    text = prompt + v['sentence']
    
    input = {'image': img, 'text': text}
    result = ofa_pipe(img)
    print(result[OutputKeys.CAPTION])
          
#     result = ofa_pipe(input)
#     reason = result[OutputKeys.TEXT][0]
#     res[img_key] = reason
# with open('./answers/' + prompt + '_onestep.json', 'w') as f:
#     json.dump( res, f)
# -


