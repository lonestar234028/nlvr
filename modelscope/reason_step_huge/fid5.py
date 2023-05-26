import data_prepare as nlvr_data

img_root = str(nlvr_data.d.img_root_abs)
json_root = nlvr_data.d.test_json_abs_path
print("img_root->", img_root)
print("json_root->", json_root)

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
current_dir = os.path.dirname(os.path.abspath(__file__))
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
parser.add_argument("--filename", default='20230226', type=str,
                    help="device gpu")
parser.add_argument("--partpre", default=0, type=str,
                    help="partpre")
parser.add_argument("--partpost", default=6967, type=str,
                    help="partpost")
args, _ = parser.parse_known_args()

if len(args.device) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = args

with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))
promptsall = []
prompt = args.prompt.replace('#',' ').replace('*','\'')
intermedia_rationals_dir = os.path.join(current_dir, 'reasons_v6_1')
for file in os.listdir(intermedia_rationals_dir):
    if file.endswith('json'):
        with open( os.path.join(intermedia_rationals_dir, file)) as f:
            prompts_tmp = json.load(f)
            promptsall.append(prompts_tmp)
print("prompts:", len(promptsall))

start = args.partpre
end = args.partpost
if isinstance(start, str):
    start = int(start)
if isinstance(end, str):
    end = int(end)

res = {}
from tqdm import tqdm
texts = []
magic = 99

for i in tqdm(range(start,2)):
    v = ann[i]
    text = ['Does it make sense:' + v['sentence']]
    # if i > magic :
    #     break
    # if i < magic:
    #     continue
    images = v['images']
    img_key = v['sentence'] + '##' + '##'.join(images)
    k1 = images[0][len('test1/') :]
    k2 = images[1][len('test1/') :]
    img1 = str(os.path.join(img_root, images[0]))
    img2 = str(os.path.join(img_root, images[1]))
    print("img,img1,img2:", img1, img2)
    for prompts in promptsall:
      
        pmts = prompts[img_key].split('##')
        if len(pmts) != 3:
            continue
        text_inner = pmts[0] + ',left image:' + pmts[1] + ', right image:' + pmts[2] \
            + '. Therefore, does it make sense:' + v['sentence']
        if (pmts[1] == 'yes' or pmts[2] == 'yes' or pmts[1] == 'no' or pmts[2] == 'no'):
            continue
        if len(pmts[2]) == 0: # caption generated 
            text_inner =  'left image:' + pmts[0] + ', right image:' + pmts[1] \
            + '. Therefore, does it make sense:' + v['sentence']
        text.append(text_inner)
    input = {'image': [img1, img2], 'text': text}
    # input = {'image': img, 'text': text}
    print("input:", input)
    print("len(text):", len(text))
    result = ofa_pipe(input)
    reason = result[OutputKeys.TEXT][0]
    res[img_key] = reason
ff =  os.path.join(current_dir, 'answers_fid_20230526' 
                   + args.filename 
                   +  '_part_'+ str(start) 
                   + '_' 
                   +  str(end) + '.json')


print("writing:",ff)
print("res:", res)
with open(ff, 'w') as f:
    json.dump( res, f)
# -


