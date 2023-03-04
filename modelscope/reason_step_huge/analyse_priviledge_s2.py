res_all = {}
res_orig = {}
with open('analyse_res_cot_1230_pri.tsv', 'r',encoding='utf-8') as file:
    for (i,line) in enumerate(file.readlines()):
        if i == 0 :
            continue
        x = line.split('\t')
        alg = x[0]
        correct = x[2]
        y =  NLVR(x[0], 
                            x[1], 
                            x[2], 
                            x[3], 
                            x[4], 
                            x[5], 
                            x[6], 
                            x[7])
        if alg == './answers/_onestep.json':
            res_orig[x[7]] = y
        if alg in res_all:
            res_all[alg][x[7]] = y
        else:
            res_all[alg] = {}
            res_all[alg][x[7]] = y
        

# Define test case class
class NLVR:
    def __init__(self, model, count, correct, falsepositive, falsenegative, predicty, predictn, imagekey):
        self.model = model
        self.count = count
        self.correct = correct
        self.falsepositive = falsepositive
        self.falsenegative = falsenegative
        self.predicty = predicty
        self.predictn = predictn
        self.imagekey = imagekey
    def __str__(self):
        return      self.model + '	' + self.count + '	' + self.correct + '	' + self.falsepositive + '	' + self.falsenegative + '	' + self.predicty + '	' + self.predictn + '	' + self.imagekey


print(len(res_orig))
print(len(res_all))

reasons_all = {}
import os
import json
for f in os.listdir('./reasons/'):
    if not f.endswith('json'):
        continue
    reason = {}
    with open('./reasons/' + f, 'r') as f:
        reason = json.load( f)
        reasons_all[f.name] = reason
#         print(f.name)
#     print("res:", len(res))
print("reasons_all:", len(reasons_all))
# for k,v in reasons_all.items():
#     print(k)
#     print(v)
#     break


# -

# +
with open('analyse_res_cot_1230_pri_s2.tsv', 'w',encoding='utf-8') as file:
    file.write('Model\tCount\tCorrect\tFalse Positive\tFalse Negative\tPredictY\tPredictN\tImage Key\tPrompt\n')
    for k,v in res_orig.items():
        if (v.correct == '0'):
            for kk,vv in res_all.items():
                reasons = {}
                if kk.replace('answers','reasons') in reasons_all:
                    reasons = reasons_all[kk.replace('answers','reasons')]
            
#                 print(len(reasons))
                
                if k in vv:
#                     for q,w in reasons.items():
#                         print(q)
#                         print(w)
                        
#                         break
                   
                    if vv[k].correct == '1':
                        reason = ""
                        if k.strip() in reasons:
#                             print(reasons[k.strip()])
                            reason = reasons[k.strip()]
                        
                        file.write(vv[k].__str__().strip() + '	' + reason + '\n')

# -


