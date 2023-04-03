# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import evaluate


# Metrics
class BleuScore:
    """Wrap sacrebleu, via metrics, with necessary postprocessing

    This metric has a dependency on sacrebleu
    """

    def __init__(self):
        self.metric = evaluate.load("sacrebleu")

    def run(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]
        result = self.metric.compute(predictions=preds, references=labels)
        return {"bleu": round(result["score"], 4)}


class RougeScore:
    """Wrap rouge calculation with necessary postprocessing

    This metric has a dependency on rouge and nltk
    """

    def __init__(self):
        import nltk
        from filelock import FileLock

        self.metric = evaluate.load("rouge")

        try:
            nltk.data.find("tokenizers/punkt")
        except (LookupError, OSError):
            with FileLock(".lock") as _:
                nltk.download("punkt", quiet=True)
        self.tok = nltk.sent_tokenize

    def run(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(self.tok(pred)) for pred in preds]
        labels = ["\n".join(self.tok(label)) for label in labels]
        result = self.metric.compute(predictions=preds, references=labels, use_stemmer=True)
        return {k: round(v * 100, 4) for k, v in result.items()}


def get_metric(name):
    metric = RougeScore() if name == "rouge" else BleuScore()
    return metric
