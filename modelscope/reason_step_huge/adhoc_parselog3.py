
import json
res = {}
with open('fid_3.log', 'r') as f:
    lines = f.readlines()
    for x in lines:
        if x.startswith("input: {'image':"):
            z = len("input: {'image':")
            y = x[z:]
            z = y.index(", 'text'")
            sentence = y[z:]
            y = (y[:z])
            y = y[2: -1]
            z = y.split(',')
            # print(z)
            z = list(map(lambda x: x.strip()[len("'/vc_data/users/taoli1/mm/nlvr/"):-1], z))
            images = z
            # print(images)
            
            sentence = (sentence)[sentence.index('Does it make sense:') + len('Does it make sense:'):]
            try:
                sentence = sentence[:sentence.index("\", \"")]

            except:
                try:
                    sentence = sentence[:sentence.index("', ")]
                    print(sentence)
                except:
                    sentence = sentence
                
            # print(sentence)
            img_key = sentence + '##' + '##'.join(images)
            print(img_key)
        if x.startswith("hello ret  in ofa_for_all_task.py: "):
            z = len("hello ret  in ofa_for_all_task.py: ")
            y = x[z:]
            z = len("{'text': ['")
            y = y[z:]
            z = len("']}")
            y = y[:z]
            # print(y)
            res[img_key] = y
with open('adhoc_log33.json','w') as f:
    json.dump(res,f)