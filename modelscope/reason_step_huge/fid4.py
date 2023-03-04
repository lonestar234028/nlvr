img_root = '/vc_data/users/taoli1/mm/nlvr/'
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
from transformers import AutoTokenizer, AutoModel


tokenizer = AutoTokenizer.from_pretrained("facebook/contriever")
fb_model = AutoModel.from_pretrained('facebook/contriever')

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
parser.add_argument("--filename", default='20230226', type=str,
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
magic = 99
def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

for i in tqdm(range(len(ann))):
    v = ann[i]
    text = ['Does it make sense:' + v['sentence']]
    text_cdd = ['dummp']
    # if i > magic :
    #     break
    # if i < magic:
    #     continue
    images = v['images']
    img_key = v['sentence'] + '##' + '##'.join(images)
    k1 = images[0][len('test1/') :]
    k2 = images[1][len('test1/') :]
    img1 = img_root + images[0]
    img2 = img_root + images[1]
    print("img,img1,img2:", img1, img2)
    fb_sentence = [v['sentence']]
    for prompts in promptsall:
      
        pmts = prompts[img_key].split('##')
        if len(pmts) != 3:
            continue
        fb_txt = 'left image:' + pmts[1] + ', right image:' + pmts[2] \
            + '. Therefore, does it make sense:' + v['sentence']
        text_inner = pmts[0] + ',left image:' + pmts[1] + ', right image:' + pmts[2] \
            + '. Therefore, does it make sense:' + v['sentence']
        if (pmts[1] == 'yes' or pmts[2] == 'yes' or pmts[1] == 'no' or pmts[2] == 'no'):
            continue
        if len(pmts[2]) == 0: # caption generated 
            fb_txt = 'left image:' + pmts[1] + ', right image:' + pmts[2] \
            + '. Therefore, does it make sense:' + v['sentence']
            text_inner =  'left image:' + pmts[0] + ', right image:' + pmts[1] \
            + '. Therefore, does it make sense:' + v['sentence']
        text_cdd.append(text_inner)
        fb_sentence.append(fb_txt)
    prompt_len = len(fb_sentence)
    if len(fb_sentence) > 2 :
        fb_inputs = tokenizer(fb_sentence, padding=True, truncation=True, return_tensors="pt")
        fb_outputs = fb_model(**fb_inputs)
        embeddings = mean_pooling(fb_outputs[0], fb_inputs['attention_mask'])
        idx2sim = {}
        for ii in range(1,prompt_len):
            idx2sim[ii] = embeddings[0] @ embeddings[ii]
        newidx2sim = sorted(idx2sim.items(), key = lambda x: x[1], reverse= True)
        print("idx2sim: ", idx2sim)
        print("newidx2sim: ", list(map(lambda x: x[0], newidx2sim)))

        topk = 2 if len(newidx2sim) > 2 else len(newidx2sim)
        newidx2sim = newidx2sim[:topk]
        for ii in newidx2sim:
            idx = ii[0]
            text.append(text_cdd[idx])
    print("text_cdd:", text_cdd)
    input = {'image': [img1, img2], 'text': text}
    print("input:", input)
    result = ofa_pipe(input)
    reason = result[OutputKeys.TEXT][0]
    res[img_key] = reason
ff = './answers_fid_four/' + args.filename + '.json'
print("writing:",ff)
print(res)
with open(ff, 'w') as f:
    json.dump( res, f)
# -


