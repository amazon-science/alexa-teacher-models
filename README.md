# Alexa Teacher Models

This is the official Alexa Teacher Model program github page.

## AlexaTM 20B

AlexaTM 20B is a 20B-Parameter sequence-to-sequence transformer model created by the Alexa Teacher Model (AlexaTM) team at Amazon. The model was trained on a mixture of Common Crawl (mC4) and Wikipedia data across 12 languages using denoising and Causal Language Modeling (CLM) tasks.

AlexaTM 20B can be used for in-context learning. "In-context learning," also known as "prompting," refers to a method for using NLP models in which no fine tuning is required per task. Training examples are provided to the model only as part of the prompt given as inference input, a paradigm known as "few-shot in-context learning." In some cases, the model can perform well without any training data at all, a paradigm known as "zero-shot in-context learning."

To learn more about the model, please read the [Amazon Science blog post](https://www.amazon.science/blog/20b-parameter-alexa-model-sets-new-marks-in-few-shot-learning) and the [paper](https://arxiv.org/abs/2208.01448).

The model is currently available for noncommercial use via SageMaker JumpStart, as described in our [AWS blog post](https://aws.amazon.com/blogs/machine-learning/alexatm-20b-is-now-available-in-amazon-sagemaker-jumpstart/). The model can be accessed using the following steps:

1. [Create](https://aws.amazon.com/premiumsupport/knowledge-center/create-and-activate-aws-account/) an AWS account if needed.
1. In your AWS account, search for `SageMaker` in the search bar and click on it.
1. Once in the SageMaker experience, create a [domain](https://docs.aws.amazon.com/sagemaker/latest/dg/gs-studio-onboard.html) and a studio user if none yet exist. All of the default settings can be used.
1. In the control panel, click `Launch app` next to the user you wish to use. Launch a studio instance.
1. Once in the studio, there will be a launcher showing JumpStart as one of the tiles. Click `Go to SageMaker Jumpstart`. Alternatively, JumpStart can be accessed by 3-pointed orange symbol on the far left of the studio.
1. Once in JumpStart, click the `Notebooks` button.
1. Browse or search for our example notebook entitled `In-context learning with AlexaTM 20B`.
1. There will be a button at the top to copy the read-only version into your studio.
1. Ensure that your kernel has started, and run the notebook.

Note: You can also find our example notebook [here](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_amazon_algorithms/jumpstart_alexatm20b/Amazon_Jumpstart_AlexaTM_20B.ipynb)

### Load the Model and Run Inference

```python
from alexa_teacher_models import AlexaTMTokenizerFast
tokenizer = AlexaTMTokenizerFast.from_pretrained('/path/to/AlexaTM-20B-pr/')


# Load the model
from alexa_teacher_models import AlexaTMSeq2SeqForConditionalGeneration
model = AlexaTMSeq2SeqForConditionalGeneration.from_pretrained('/path/to/AlexaTM-20B-pr/')
```

You can also use the `AutoTokenizer` and `AutoModelForSeq2SeqLM` as you would in any other HuggingFace Transformer
program by importing `alexa_teacher_models`:

```python
import alexa_teacher_models
...
tokenizer = AutoTokenizer.from_pretrained('/path/to/AlexaTM-20B-pr/')
model = AutoModelForSeq2SeqLM.from_pretrained('/path/to/AlexaTM-20B-pr/')

```

Load the model on 4 gpus:

```python
model.bfloat16()
model.parallelize(4)
```

Run the model in CLM mode:
```python
# qa
test = """[CLM] Question: Who is the vocalist of coldplay? Answer:"""
print('Input:', test)
encoded = tokenizer(test, return_tensors="pt").to('cuda:0')
generated_tokens = model.generate(input_ids=encoded['input_ids'],
                                  max_length=32,
                                  num_beams=1,
                                  num_return_sequences=1,
                                  early_stopping=True)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
```

Run the model in denoising mode:
```python
# denoising
test = "we went to which is the capital of France"
print('Input:', test)
encoded = tokenizer(test, return_tensors="pt").to('cuda:0')
generated_tokens = model.generate(input_ids=encoded['input_ids'],
                                  max_length=32,
                                  num_beams=5,
                                  num_return_sequences=5,
                                  early_stopping=True)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
```

## Running the repl example

A sample Read Execute Print Loop (REPL) program is provided in the samples.  It can be used to interact with
any AlexaTM model, and has a flexible set of command line arguments, including support for sampling and using multiple turns of history as context

```
$ pip install alexa_teacher_models[repl]
$ python -m alexa_teacher_models.scripts.repl --model /path/to/AlexaTM-20B-pr/ --max_length 64
$ python -m alexa_teacher_models.scripts.repl --model /path/to/AlexaTM-20B-pr/ --max_length 64 --do_sample --max_history 3 --join_string " </s> "

```

## Fine-tuning with DeepSpeed on a single P4

*Note* We strongly recommend training on multiple instances.  For information on how to do this, see the section below

To run on a single P4 (8 GPUs), you will need to use CPU offload.  A deepspeed config is provided in the `scripts/deepspeed` directory.
Assuming you have a training and validation JSONL formatted file, a run would look like this:
```
$ pip install alexa_teacher_models[ft]
$ deepspeed --num_gpus 8 --module alexa_teacher_models.scripts.finetune --per_device_train_batch_size $BS \
    --deepspeed deepspeed/zero3-offload.json \
    --model_name_or_path /home/ubuntu/AlexaTM/ --max_length 512 --bf16 --output_dir output \
    --max_target_length 64 --do_train --learning_rate 1e-7 \
    --train_file train.json --validation_file valid.json \
    --num_train_epochs 1 --save_steps 1000


```

## Fine-tuning with DeepSpeed on multiple machines

There is a [detailed tutorial](docs/EFA.md) demonstrating how to fine-tune 20B across multiple machines in EC2 using [Elastic Fabric Adapter (EFA)](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa.html).

## Citation
If you use AlexaTM 20B, please use the following BibTeX entry.

```
@article{soltan2022alexatm,
  title={AlexaTM 20B: Few-Shot Learning Using a Large-Scale Multilingual Seq2seq Model},
  author={Saleh Soltan, Shankar Ananthakrishnan, Jack FitzGerald, Rahul Gupta, Wael Hamza, Haidar Khan, Charith Peris, Stephen Rawls, Andy Rosenbaum, Anna Rumshisky, Chandana Satya Prakash, Mukund Sridhar, Fabian Triefenbach, Apurv Verma, Gokhan Tur, Prem Natarajan},
  year={2022}
}
```


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License
The code in this package is subject to [License](LICENSE). However, 
the model weights are subject to [Model License](MODEL_LICENSE.md).
