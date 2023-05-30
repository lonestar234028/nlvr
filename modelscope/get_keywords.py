from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, *args, **kwargs):
        super().__init__(
            model=AutoModelForTokenClassification.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model),
            *args,
            **kwargs
        )

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs,
            aggregation_strategy=AggregationStrategy.SIMPLE,
        )
        return np.unique([result.get("word").strip() for result in results])
# Load pipeline
model_name = "ml6team/keyphrase-extraction-kbir-inspec"
extractor = KeyphraseExtractionPipeline(model=model_name)
# Inference
lines = []
with open('sentences.txt', 'r') as f:
    lines = f.readlines()

kpsall = {}
from tqdm import tqdm
for text in tqdm(lines):
    text = text.replace("\n", "")
    keyphrases = extractor(text)
    kpsall[text] = keyphrases.tolist()
    # break

print(kpsall)
import json
with open('kpsall.json', 'w') as f:
    json.dump(kpsall, f)

