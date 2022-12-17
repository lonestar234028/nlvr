from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
from transformers.models.ofa.generate import sequence_generator

import json

img_root = '/vc_data/users/taoli1/mm/nlvr/'
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


def get_patch(imagepath):
    img = Image.open( img_root + imagepath)
    patch_img = patch_resize_transform(img).unsqueeze(0).to(device)
    return patch_img


# +
res = {}
prompt = args.prompt.replace('#',' ').replace('*','\'')

i = 0
from tqdm import tqdm
for i in tqdm(range(len(ann))):
    # if i > 0:
    #     break
    v = ann[i]
    reason = prompt + '##'
    images = v['images']
    img_key = v['sentence'] + '##' + '##'.join(images)
    patch_img = get_patch(images[0])
    reason0 = get_reason(prompt, patch_img)
    patch_img = get_patch(images[1])
    reason1 = get_reason(prompt, patch_img)
    reason += reason0 + '##' + reason1
    res[img_key] = reason
with open('reasons/' + prompt + '.json', 'w') as f:
    json.dump(res, f)  
