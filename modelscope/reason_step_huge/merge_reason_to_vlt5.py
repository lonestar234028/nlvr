json_root = '/home/taoli1/VL-T5/datasets/nlvr/backup/test.json'

import json,os

ann = []

with open(json_root, 'r') as f:
    ann = json.load(f)

print("ann:",len(ann))
print("ann:",ann[0])
res_all = {}
import copy
for f in os.listdir('./reasons/'):
    res = {}
    name = f
    if not f.endswith('json'):
        continue
    print(f)
    with open('./reasons/' + f, 'r') as f:
        res = json.load( f)
        res_all[f.name] = res
    res_new = {}
    for k,v in res.items():
        s = k.split('##')
        k1 = s[1][s[1].find('/') + 1:len(s[1]) - len('.png')]
        k2 = s[2][s[2].find('/') + 1:len(s[2]) - len('.png')]
        new_k = k1 + '##' + k2
        res_new[new_k] = v
    new_ann = []
    for item1 in ann:
        item = copy.deepcopy(item1)
        k =  item['img0'] + '##' + item['img1']
        pmts = res_new[k].split('##')
        if len(pmts) != 3:
            continue
        p1 = pmts[1]
        p2 = pmts[2]
        if 'caption' in f.name:
            p1 = pmts[0]
            p2 = pmts[1]
        if 'yes' in p1 or 'yes' in p2 or 'no' in p1 or 'no' in p2:
            new_ann.append(item)
            continue
        if 'caption' in f.name:
            p1 = pmts[0]
            p2 = pmts[1]
            text = 'left image:' + p1 + ', right image:' + p2 \
            + '. Therefore, does it make sense:' + item['sent']
            item['sent'] = text
            new_ann.append(item)
            continue 
        text = pmts[0] + ' left image:' + pmts[1] + ', right image:' + pmts[2] \
            + '. Therefore, does it make sense:' + item['sent']
        item['sent'] = text
        new_ann.append(item)
    
    with open('./prompted/' + name, 'w') as f:
        json.dump( new_ann, f)

    print("new_ann:",len(new_ann))    
    print("new_ann:",new_ann[100])

print("res_all:",len(res_all))