from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.multi_modal import OfaPreprocessor
model = 'damo/ofa_visual-question-answering_pretrain_large_en'
preprocessor = OfaPreprocessor(model_dir=model)
ofa_pipe = pipeline(
    Tasks.visual_question_answering,
    model=model,
    preprocessor=preprocessor)
image = 'https://xingchen-data.oss-cn-zhangjiakou.aliyuncs.com/maas/visual-question-answering/visual_question_answering.png'
text = 'what is grown on the plant?'
input = {'image': image, 'text': text}
result = ofa_pipe(input)
print(result[OutputKeys.TEXT]) # ' money'