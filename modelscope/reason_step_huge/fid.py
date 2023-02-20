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
parser.add_argument("--filename", default='20230125', type=str,
                    help="device gpu")

args, _ = parser.parse_known_args()

if len(args.device) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args

with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))
promptsall = []
prompt = args.prompt.replace('#',' ').replace('*','\'')
for file in os.listdir("./reasons/"):
    if file.endswith('json'):
        with open('./reasons/' + file) as f:
            prompts_tmp = json.load(f)
            promptsall.append(prompts_tmp)
print("prompts:", len(promptsall))


res = {}
from tqdm import tqdm
texts = []
for i in tqdm(range(len(ann))):
    v = ann[i]
    text = ['Does it make sense:' + v['sentence']]
    # if i > 0:
    #     break
    images = v['images']
    img_key = v['sentence'] + '##' + '##'.join(images)
    k1 = images[0][len('test1/') :]
    k2 = images[1][len('test1/') :]
    img = img_root + k1 + "-"+ k2
    print("img:",img)
    for prompts in promptsall:
        pmts = prompts[img_key].split('##')
        if len(pmts) != 3:
            continue
        text_inner = pmts[0] + ',left image:' + pmts[1] + ', right image:' + pmts[2] \
            + '. Therefore, does it make sense:' + v['sentence']
        if (pmts[1] == 'yes' or pmts[2] == 'yes' or pmts[1] == 'no' or pmts[2] == 'no'):
            continue
        text.append(text_inner)
    input = {'image': img, 'text': text}
    print("input:", input)
    result = ofa_pipe(input)
    reason = result[OutputKeys.TEXT][0]
    res[img_key] = reason
with open('./answers_fid/' + args.filename + '.json', 'w') as f:
    json.dump( res, f)
# -


