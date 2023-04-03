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

import logging
from dataclasses import dataclass, field
from typing import Optional
from datasets import DatasetDict
from alexa_teacher_models.scripts.train_utils import load_data, create_tokenizer, Preprocessor

# This allows us to us Auto* from HuggingFace to get to the model
import alexa_teacher_models

from transformers import (
    HfArgumentParser,
)

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    vocab_file: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer vocab file. Defaults to 20b_tokenizer.model"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a JSONL or CSV)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on " "a JSONL file."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a JSONL file."},
    )
    output_dir: str = field(
        default=None,
        metadata={"help": "An output directory to save the arrow files to"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={"help": "A prefix to add before every source text.  This should be [CLM] for AlexaTM LM tasks"},
    )

    source_field: Optional[str] = field(
        default="x",
        metadata={"help": "The field to get the source data from"},
    )

    target_field: Optional[str] = field(
        default="y",
        metadata={"help": "The field to get the target data from"},
    )

    allow_load_from_cache: bool = field(
        default=False,
        metadata={"help": "Allow loading a dataset from the cache"},
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()

    datasets = load_data(
        data_args.train_file,
        data_args.validation_file,
        data_args.test_file,
        data_args.dataset_name,
        data_args.dataset_config_name,
        model_args.cache_dir,
        model_args.use_auth_token,
    )
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if 'train' in datasets:
        column_names = datasets["train"].column_names
    elif 'validation' in datasets:
        column_names = datasets["validation"].column_names
    elif 'test' in datasets:
        column_names = datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    logger.info(column_names)

    tokenizer = create_tokenizer(model_args.model_name_or_path, model_args.vocab_file, model_args.cache_dir)
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    preprocessor = Preprocessor(
        tokenizer,
        column_names,
        data_args.max_length,
        max_target_length,
        data_args.source_field,
        data_args.target_field,
        prefix,
        padding,
        data_args.ignore_pad_token_for_loss,
        num_workers=data_args.preprocessing_num_workers,
        load_from_cache_file=data_args.allow_load_from_cache,
    )

    proc_datasets = {}
    if "train" in datasets:
        train_dataset = datasets["train"]
        train_dataset = preprocessor.run(
            train_dataset,
            desc="Running tokenizer on train dataset",
        )
        proc_datasets['train'] = train_dataset

    if "validation" in datasets:
        eval_dataset = datasets["validation"]
        eval_dataset = preprocessor.run(
            eval_dataset,
            desc="Running tokenizer on validation dataset",
        )
        proc_datasets['validation'] = eval_dataset
    if "test" in datasets:
        predict_dataset = datasets["test"]
        predict_dataset = preprocessor.run(
            predict_dataset,
            desc="Running tokenizer on prediction dataset",
        )
        proc_datasets['test'] = predict_dataset

    proc_datasets_dict = DatasetDict(proc_datasets)
    proc_datasets_dict.save_to_disk(data_args.output_dir)


if __name__ == "__main__":
    main()
