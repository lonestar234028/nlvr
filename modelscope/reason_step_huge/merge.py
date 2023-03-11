import os

kkv = {}

for file in os.listdir("./reasons_v5/"):
    if file.endswith('json'):
        x = file.split('_part_')
        y = x[0].strip()
        kkv[y] =  kkv[y] + 1 if y in kkv else 1
ok ={}
for k,v in kkv.items():
    ok[k] = {}

import json
for file in os.listdir("./reasons_v5/"):
    if file.endswith('json'):
        x = file.split('_part_')
        y = x[0].strip()
        with open(os.path.join("./reasons_v5/", file), 'r') as f:
            z = json.load(f)
            ok[y].update(z)
for k,v in ok.items():
    # if len(v) == 6967:
    with open(os.path.join('./reasons_v77', k + '.json'), 'w') as f:
        json.dump(v, f)
    print(k, len(v))