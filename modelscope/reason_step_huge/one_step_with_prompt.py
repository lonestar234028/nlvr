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
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor

# -

model = 'damo/ofa_visual-question-answering_pretrain_huge_en'
# model = 'damo/ofa_pretrain_huge_en'
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

args, _ = parser.parse_known_args()

if len(args.device) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args

# +
with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))

prompt = args.prompt.replace('#',' ').replace('*','\'')
# -

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
    print("input:",input)
    result = ofa_pipe(input)
    reason = result[OutputKeys.TEXT][0]
    res[img_key] = reason
with open('./answers/' + prompt + '_onestep.json', 'w') as f:
    json.dump( res, f)


