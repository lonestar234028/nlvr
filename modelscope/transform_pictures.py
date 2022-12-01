img_root = '/vc_data/users/taoli1/mm/nlvr/'
json_root = '/vc_data/users/taoli1/mm/finetune/nlvr_test.json'
ann = {}
import json
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
resolution = 480
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean, std),
    ])
with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))
i = 0
for v in ann:
    # if i > 0:
    #     break
    images = v['images']
    print("images:",images)
    img1 = Image.open(img_root + images[0])    
    img2 = Image.open(img_root + images[1])
    patch_img1 = patch_resize_transform(img1)
    patch_img2 = patch_resize_transform(img2)
    patch_img = torch.cat((patch_img1, patch_img2), dim=-1)
    k1 = images[0][len('test1/') :]
    k2 = images[1][len('test1/') :]
    save_image(patch_img, "imagesv1/" + k1 + "-"+ k2)
    i += 1
