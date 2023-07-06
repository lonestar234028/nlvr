result = []
result_dict = {}
with open('gptv.log', 'r') as f:
    for line in f.readlines():
        if not line.__contains__('question_id'):
            continue
        single_res = {}
        line = line[line.find('question_id'):]
        res = line.strip().split(',')
        # print(res)
        if len(res) < 4:
            continue
        ques = res[0].split(':')
        single_res.update({ques[0]: int(ques[1])})
        ans = res[3].split(':')
        answord = ans[1]
        if len(res) > 4:
            answord = ','.join(res[3:])
        single_res.update({ans[0]: answord})
        result.append(single_res)
        result_dict[ques[1]] = single_res
# print("result", result)
print("len(result)", len(result))
print("len(result_dict)", len(result_dict))

def get_res_from_dict(qid):
    # print("result_dict", result_dict)
    if qid in result_dict:
        return result_dict[qid]
    else:
        return None
