
ann = {}
import json,os,sys
import argparse
import jsonlines

lines = []
for f in os.listdir('./analysis/'):
    if '20221231' not in f:
        continue
    try:
        with open('./analysis/' + f, 'r') as f:
            lines.extend(f.readlines())
    except:
        continue
# +
with open('analyse_res_cot_merge_20221231.tsv', 'w',encoding='utf-8') as file:
    file.writelines(lines)