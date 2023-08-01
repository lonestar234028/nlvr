json_root = '/vc_data/users/taoli1/mm/finetune/nlvr_test.json'

ann = {}
import json,os,sys
import argparse

with open(json_root, 'r') as f:
    ann = json.load(f)
print("ann:", len(ann))
res_all = {}
i = 0
res = {}
fname = '/vc_data/users/taoli1/nlvrresearch/modelscope/reason_step_huge/answers_fid_20230714/top13_part_0_6967.json'
# with open('./answers_fid_two/' + f, 'r') as f: # best
# with open('./answers_fid/' + f, 'r') as f:
# with open('./answers_fid_three/' + f, 'r') as f:
with open(fname, 'r') as f:

    res = json.load( f)
print("res:", len(res))

# with open('analyse_res_cot_fid_20230127.tsv', 'w',encoding='utf-8') as file:
# with open('analyse_res_cot_fid_three_20230204.tsv', 'w',encoding='utf-8') as file:
with open('analyse_res_cot_fid_meanfive_retrieve_20230715_top13.tsv', 'w',encoding='utf-8') as file:
    file.write('Model\tCount\tCorrect\tFalse Positive\tFalse Negative\tPredictY\tPredictN\tAcc\tCoverage\n')
    all = 0
    correct = 0
    fp = 0
    fn = 0
    pred_Y = 0
    pred_N = 0
    acc = 0
    coverate = 0
    for i in range(len(ann)):

        v = ann[i]
        images = v['images']
        img_key = v['sentence'] + '##' + '##'.join(images)
        r = res[img_key].strip()
#             if i == 4:
#                 print(img_key)
        
        if r == 'yes':
            all += 1
            pred_Y += 1
            if v['label'] == 'True':
                correct += 1
            else:
                fp += 1
        elif  r == 'no':
            all += 1
            pred_N += 1
            if v['label'] == 'False':
                correct += 1
            else: 
                fn += 1
    file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2%}\t{:.2%}\n".format('FiD_2Patch_meanpooling',all,correct,fp,fn,pred_Y,pred_N,correct/6967,all/6967))



