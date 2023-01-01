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

model = 'damo/ofa_visual-question-answering_pretrain_large_en'
preprocessor = OfaPreprocessor(model_dir=model)
ofa_pipe = pipeline(
    Tasks.visual_question_answering,
    model=model,
    preprocessor=preprocessor)

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default='First,', type=str,
                    help="prompt")
parser.add_argument("--device", default='', type=str,
                    help="device gpu")

parser.add_argument("--lang", default='ta', type=str,
                    help="lang")
args, _ = parser.parse_known_args()

if len(args.device) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args
from dataset import *
ann = load_annotations(args.lang)
print("ann:", len(ann))

prompts = {}
prompt = args.prompt.replace('#',' ').replace('*','\'')


lang =  args.lang
with open('reasons/' + lang +'_' + prompt + '.json', 'r') as f:
    prompts = json.load(f)
print("prompts:", len(prompts))

res = {}
from tqdm import tqdm
for i in tqdm(range(len(ann))):
    # if i > 0:
    #     break
    v = ann[i]
    img_key = v['image_id_0'] + '##' + v['image_id_1']
    pmts = prompts[img_key].split('##')
    if len(pmts) != 3:
        continue
    print("pmts:", pmts)
    img = "images/" + lang + '/' + img_key + '.jpg'
    text = pmts[0] + ' left image:' + pmts[1] + ', right image:' + pmts[2] \
        + '. Therefore, does it make sense:' + v['sentence']
    if i == 0:
        print("img:", img)
        print("text:", text)
    input = {'image': img, 'text': text}
    result = ofa_pipe(input)
    reason = result[OutputKeys.TEXT][0]
    res[img_key] = reason

with open('./answers/' + lang + '-' + prompt + '.json', 'w') as f:
    json.dump( res, f)


