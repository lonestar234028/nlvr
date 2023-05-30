# img_root = '/home/taoli1/nlvr/modelscope/images/'
img_root = '/vc_data/users/taoli1/mm/nlvr/'

json_root = '/vc_data/users/taoli1/mm/finetune/nlvr_test.json'
ann = {}
import json,os,sys
with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))

sentence_set = set()

for item in ann:
    sentence_set.add(item['sentence'])
    
with open('sentences.txt', 'w') as f:
    for line in sentence_set:
        f.write(line + '\n')