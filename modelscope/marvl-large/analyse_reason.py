# json_root = '/vc_data/users/taoli1/mm/finetune/nlvr_test.json'
json_root = '/home/taoli1/marvl-code-forked/data/zh/annotations_machine-translate/marvl-zh_gmt.jsonl'

ann = {}
import json,os,sys
import argparse
import jsonlines

def load_annotations():
    items = []
    with jsonlines.open(json_root) as reader:
        # Build an index which maps image id with a list of hypothesis annotations.
        count = 0
        for annotation in reader:
            dictionary = {}
            dictionary["image_id_0"] = annotation["left_img"].split("/")[-1].split(".")[0]
            dictionary["image_id_1"] = annotation["right_img"].split("/")[-1].split(".")[0]
            dictionary["question_id"] = count

            dictionary["sentence"] = str(annotation["caption"])
            dictionary["labels"] = [int(annotation["label"])]
            dictionary["label"] = annotation["label"]
            dictionary["concept"] = str(annotation["concept"])
            dictionary["scores"] = [1.0]
            dictionary["ud"] = str(annotation["id"])
            items.append(dictionary)
            count += 1
            if count < 2:
                print("loading_annotations: ")
                print(dictionary)
    return items

ann = load_annotations()
print("ann:", len(ann))
res_all = {}
i = 0
prompt = 'First,'
for f in os.listdir('./answers/'):
    res = {}
    with open('./answers/' + f, 'r') as f:
        res = json.load( f)
        res_all[f.name] = res
        print(f.name)
#     print("res:", len(res))
print("res_all:", len(res_all))

weight = {}
weight["./answers/_onestep.json"] = 1
weight["./answers/Does it make sense:_onestep.json"] = 1
weight["./answers/nlvr:_onestep.json"] = 1
weight["./answers/vqa:_onestep.json"] = 1
weight["./answers/Am I making sense:_onestep.json"] = 1
# weight["./answers/Let's solve this problem by splitting it into steps..json"] = 1
# weight["./answers/The answer is after the proof..json"] = 1
# weight["./answers/Let's think about this logically..json"] = 1
# weight["./answers/Let's think like a detective step by step..json"] = 1
# weight["./answers/Let's be realistic and think step by step..json"] = 1
# weight["./answers/First,.json"] = 1
# weight["./answers/Let's think step by step..json"] = 1

# +
with open('analyse_res_cot.tsv', 'w',encoding='utf-8') as file:
    file.write('Model\tCount\tCorrect\tFalse Positive\tFalse Negative\tPredictY\tPredictN\tAcc\tCoverage\n')
    res_ensemble = {}
    for p, res in res_all.items():
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
            img_key = v['image_id_0'] + '##' + v['image_id_1']    
            r = res[img_key].strip()
#             if i == 4:
#                 print(img_key)
            if r == 'yes' or r == 'no':
                if img_key in res_ensemble:
                    if r in res_ensemble[img_key]:
                        res_ensemble[img_key][r] += weight[p]
                    else: 
                        res_ensemble[img_key][r] = weight[p]
                else:
                    res_ensemble[img_key] = {r:weight[p]}
            if r == 'yes':
                all += 1
                pred_Y += 1
                if v['label'] == True:
                    correct += 1
                else:
                    fp += 1
            elif  r == 'no':
                all += 1
                pred_N += 1
                if v['label'] == False:
                    correct += 1
                else: 
                    fn += 1
        file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2%}\t{:.2%}\n".format(p,all,correct,fp,fn,pred_Y,pred_N,correct/len(ann),all/len(ann)))
        print("prompt:",p)
        print("all:",all)
        print("correct:",correct)
        print("fp:",fp)
        print("fn:",fn)
        print("pred_Y:",pred_Y)
        print("pred_N:",pred_N)
        print("acc:",correct / len(ann))
        print("coverate:", all / len(ann))
        print("+++++++++++++")
    i = 0
    all = 0
    correct = 0
    fp = 0
    fn = 0
    pred_Y = 0
    pred_N = 0
    acc = 0
    coverate = 0
    res_es = {}
    for k,v in res_ensemble.items():
        r = sorted(v, key=lambda key_value: key_value[1], reverse=True)[0]
        y = sorted(v, key=lambda key_value: key_value[1], reverse=True)[0]

        if i == 1099:
            print(v)
            print( sorted(v.items(), key=lambda key_value: key_value[1], reverse=True))
        res_es[k] = y
        i += 1
    print("res_es:",len(res_es))

    for i in range(len(ann)):

            v = ann[i]
            img_key = v['image_id_0'] + '##' + v['image_id_1']
            if img_key not in res_es:
                continue
            r = res_es[img_key].strip()
            if i == 4:
                print(img_key)

            if r == 'yes':
                all += 1
                pred_Y += 1
                if v['label'] == True:
                    correct += 1
                else:
                    fp += 1
            elif  r == 'no':
                all += 1
                pred_N += 1
                if v['label'] == False:
                    correct += 1
                else: 
                    fn += 1
            else :
                print(k)
                print(r)
                
    file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2%}\t{:.2%}\n".format('Ensemble',all,correct,fp,fn,pred_Y,pred_N,correct/len(ann),all/len(ann)))
    print("prompt:",'Ensemble')
    print("all:",all)
    print("correct:",correct)
    print("fp:",fp)
    print("fn:",fn)
    print("pred_Y:",pred_Y)
    print("pred_N:",pred_N)
    print("acc:",correct / len(ann))
    print("coverate:", all / len(ann))
    print("+++++++++++++")    
        
#     print(res_ensemble['The right image shows three bottles of beer lined up.##test1/test1-0-0-img0.png##test1/test1-0-0-img1.png'])

#     i += 1
#     k1 = images[0][len('test1/') :]
#     k2 = images[1][len('test1/') :]
#     img = img_root + k1 + "-"+ k2
#     text = pmts[0] + ' left image:' + pmts[1] + ', right image:' + pmts[2] \
# + '. Therefore, does it make sense:' + v['sentence']
    
#     input = {'image': img, 'text': text}
#     result = ofa_pipe(input)
#     reason = result[OutputKeys.TEXT][0]
#     res[img_key] = reason

# -


