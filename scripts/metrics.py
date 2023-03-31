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
