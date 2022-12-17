from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator

import json

img_root = '/home/taoli1/nlvr/modelscope/images/'
json_root = '/vc_data/users/taoli1/mm/finetune/nlvr_test.json'
with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", default="Firstly,", type=str,
                    help="prompt")
parser.add_argument("--device", default='cpu', type=str,
                    help="device gpu")
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480
args, _ = parser.parse_known_args()
device = args.device


patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(), 
    transforms.Normalize(mean=mean, std=std)
])

ckpt_dir='OFA-Sys/ofa-huge'
tokenizer = OFATokenizer.from_pretrained(ckpt_dir)


model = OFAModel.from_pretrained(ckpt_dir, use_cache=False).to(device)

generator = sequence_generator.SequenceGenerator(
    tokenizer=tokenizer,
    beam_size=5,
    max_len_b=16,
    min_len=0,
    no_repeat_ngram_size=3,
)
import torch


def get_reason(prompt, patch_img):
    data = {}
    inputs = tokenizer([prompt], return_tensors="pt").input_ids.to(device)
    data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}
    gen_output = generator.generate([model], data)
    gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]
    reason = (tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip())
    return reason


def get_patch(images):
    k1 = images[0][len('test1/') :]
    k2 = images[1][len('test1/') :]
    img = img_root + k1 + "-"+ k2
    imgopened = Image.open( img )
    patch_img = patch_resize_transform(imgopened).unsqueeze(0).to(device)
    return patch_img


# +
res = {}
prompts = {}
prompt = args.prompt.replace('#',' ').replace('*','\'')
with open('/home/taoli1/nlvr/ofa/reasons/' + prompt + '.json', 'r') as f:
    prompts = json.load(f)
i = 0
from tqdm import tqdm
for i in tqdm(range(len(ann))):
    # if i > 0:
    #     break
    v = ann[i]
    reason = prompt + '##'
    images = v['images']
    img_key = v['sentence'] + '##' + '##'.join(images)
    pmts = prompts[img_key].split('##')
    if len(pmts) != 3:
        continue
    text = pmts[0] + ' left image:' + pmts[1] + ', right image:' + pmts[2] \
        + '. Therefore, does it make sense:' + v['sentence']
    if (pmts[1] == 'yes' or pmts[2] == 'yes' or pmts[1] == 'no' or pmts[2] == 'no'):
        text = 'Does it make sense:' + v['sentence']
    patch_img = get_patch(images)
    print(text)
    reason = get_reason(text, patch_img)
    res[img_key] = reason
with open('answers/' + prompt + '.json', 'w') as f:
    json.dump(res, f)  
