import json, os, sys
from collections import namedtuple
import time
"""
ok_vqa_test:
model: ofa
evaluation_measure: https://visualqa.org/evaluation
evaluation_demo:https://github.com/GT-Vision-Lab/VQA/blob/master/PythonEvaluationTools/vqaEvalDemo.py
                modified(py2 -> py3) version in  submodule(nlvr/vqa):           
                nlvr/vqa/PythonEvaluationTools/vqaEvalDemo.py
leader_board: https://okvqa.allenai.org/leaderboard.html
"""
# update to actual value before running
# expected directoy structure
# data_sets/ok_vqa/ 
#    train2014/*.jpg
#    val2014/*.jpg
#    mscoco_train2014_annotations.json
#    mscoco_val2014_annotations.json
#    OpenEnded_mscoco_train2014_questions.json
#    OpenEnded_mscoco_val2014_questions.json
okvqa_path = "/vc_data/users/taoli1/mm/okvqa/annotations/"

test_question_json_path = "OpenEnded_mscoco_val2014_questions.json"
train_question_json_path = "OpenEnded_mscoco_train2014_questions.json"

test_annotations_json_path = "mscoco_val2014_annotations.json"
train_annotations_json_path = "mscoco_train2014_annotations.json"

test_pictures_path = "val2014"
train_pictures_path = "train2014"
test_picture_file_name_pattern = "COCO_val2014_{pic_name}"
train_picture_file_name_pattern = "COCO_train2014_{pic_name}"
okvqa_path_img = "/vc_data/users/taoli1/mm/okvqa/images/"

def pic_path_pattern(pat_str):
    def p(image_id, image_dir):
        # "COCO_val2014_000000000164.jpg"
        id = ["0"] * len("000000000164")
        image_id = str(image_id)
        l = len(image_id)
        l0 = len(id)
        assert l  < l0, "image id > max"
        for i,c in enumerate(image_id):
            id[l0 - l + i] = str(c)
        image_id = "".join(id) + ".jpg"
        return os.path.join(image_dir, pat_str.format(pic_name = image_id))
    
    return p

# Declaring namedtuple()
ok_meta = namedtuple('OK_VQA_DATA_META', ['quesion', 'annotation', 'pic_path', 'pic_name_pattern_trans_fun','test_set_name'])

"""{"image_id": 297147, "question": "What sport can you use this for?", "question_id": 2971475}"""
""" 
        {
          "answer_id": 1,
          "raw_answer": "racing",
          "answer_confidence": "yes",
          "answer": "race"
        }
"""
class Question(object):
    def __init__(self, json_q, json_a, image_path_transfer, image_dir) -> None:
        self.question_text = json_q["question"]
        self.image_id = json_q["image_id"]
        self.question_id = json_q["question_id"]
        self.image_abs_path = image_path_transfer(self.image_id, image_dir)
        self.answers = [i["raw_answer"] for i in json_a["answers"]]
        assert self.question_id == json_a["question_id"] and self.image_id == json_a["image_id"] , "OK-VQA questions-and-annotations sequence corrupted!"
        


test_path = ok_meta(os.path.join(okvqa_path, test_question_json_path), os.path.join(okvqa_path, test_annotations_json_path), 
             os.path.join(okvqa_path_img, test_pictures_path), pic_path_pattern(test_picture_file_name_pattern), "test2014")

train_path = ok_meta(os.path.join(okvqa_path, train_question_json_path), os.path.join(okvqa_path, train_annotations_json_path), 
             os.path.join(okvqa_path_img, train_pictures_path), pic_path_pattern(train_picture_file_name_pattern),"train2014")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor

# -
current_dir = os.path.dirname(os.path.abspath(__file__))
bt_sz = 10
model = 'damo/ofa_visual-question-answering_pretrain_huge_en'
preprocessor = OfaPreprocessor(model_dir=model)
ofa_pipe = pipeline(
    Tasks.visual_question_answering,
    model=model,
    preprocessor=preprocessor,
    batch_size = bt_sz)

demo_result = ofa_pipe({'image':'http://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/visual_question_answering.png','text':'what is grown on the plant?'})
print("smoke test demo_result", demo_result)
"""
results = [result]

result{
"question_id": int,
"answer": str
}
"""
res = []
from tqdm import tqdm

for d in tqdm([train_path]):
    with open(d.quesion, 'r') as f, open(d.annotation, 'r') as f2:
        questions_js = json.load(f)
        questions = list(questions_js["questions"])
        question_anns_js = json.load(f2)
        questions_anns = list(question_anns_js["annotations"])
        count_q = 0
        q_ann_pairs = list(zip (questions, questions_anns))
        num = len(q_ann_pairs)
        for i in tqdm(range(0, num, bt_sz)):
            q_ann_js_batch = q_ann_pairs[i: min(i + bt_sz ,num)]
            # print(q_js, a_js)
            input_batch = []
            output_batch = []
            for q_js, a_js in q_ann_js_batch: 
                q = Question(q_js, a_js, d.pic_name_pattern_trans_fun, d.pic_path)
                count_q+=1
                q_and_a = {}
                q_and_a.update({"question_id":int(q.question_id)})
                input_batch.append({'image': q.image_abs_path, 'text': q.question_text})
                output_batch.append(q_and_a)
            answer_batch = ofa_pipe(input_batch)
            for a, p in zip(answer_batch, output_batch):
                p.update({"answer" :a[OutputKeys.TEXT][0]})
                res.append(p)
            # if(count_q > 2):break
        print("finished questions' num:", count_q)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    t = time.localtime()
    file_name = "ofa_ok_vqa_" + d.test_set_name + "_" + "_".join([str(t.tm_year), str(t.tm_mon), str(t.tm_mday), str(t.tm_hour), str(t.tm_min), str(t.tm_sec)])
    file_path = os.path.join(current_dir, file_name)
    print("dumping into ", file_path)
    with open(file_path, 'w') as output_f:
        json.dump(res, output_f)

    
    
    
