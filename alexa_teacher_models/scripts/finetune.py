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

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import is_torch_bf16_gpu_available
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from alexa_teacher_models.scripts.train_utils import load_data, create_tokenizer, Preprocessor, setup_logging
from alexa_teacher_models.scripts.metrics import get_metric

# This allows us to us Auto* from HuggingFace to get to the model
import alexa_teacher_models

from transformers import (
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
    Seq2SeqTrainer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
)

logger = logging.getLogger(__name__)

# Default to wandb off unless they explicitly have it set
if "WANDB_DISABLED" not in os.environ:
    os.environ["WANDB_DISABLED"] = "true"

# It is possible to run bfloat16 on V100s, but the HF Seq2SeqTrainingArguments won't allow it
if not is_torch_bf16_gpu_available():
    from alexa_teacher_models.scripts.train_utils import VoltaBFloat16Seq2SeqTrainingArguments as Seq2SeqTrainingArguments
else:
    from transformers import Seq2SeqTrainingArguments


def create_model(model_name_or_path, vocab_size, config_name=None, cache_dir=None, use_auth_token=False):
    """Create the model"""
    config = AutoConfig.from_pretrained(
        config_name if config_name else model_name_or_path,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
        use_auth_token=True if use_auth_token else None,
    )

    model.resize_token_embeddings(vocab_size)
    return model


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
    vocab_file: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer vocab file. Defaults to 20b_tokenizer.model"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
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
        default=None, metadata={"help": "The input training data file (a JSONL, CSV or a preprocessed arrow file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics on (a JSONL, CSV or a preprocessed arrow file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to predict metrics on a file. (a JSONL, CSV or a preprocessed arrow file)"
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
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
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    eval_metric: Optional[str] = field(
        default="bleu", metadata={"help": "What metric to use.  Options are `bleu` and `rouge`"}
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    is_preproc: bool = field(
        default=False,
        metadata={
            "help": "Whether the data is already preprocessed or not.  If so, we will use ``datasets.load_from_disk``"
        },
    )
    source_prefix: Optional[str] = field(
        default="",
        metadata={"help": "A prefix to add before every source text.  This should be \"[CLM] \" for AlexaTM LM tasks"},
    )

    source_field: Optional[str] = field(
        default="x",
        metadata={"help": "The field to get the source data from"},
    )

    target_field: Optional[str] = field(
        default="y",
        metadata={"help": "The field to get the target data from"},
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None and not self.is_preproc:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None and not self.is_preproc:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    setup_logging(logger, training_args.get_process_log_level())

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = last_available_checkpoint(training_args)

    datasets = load_data(
        data_args.train_file,
        data_args.validation_file,
        data_args.test_file,
        data_args.dataset_name,
        data_args.dataset_config_name,
        model_args.cache_dir,
        model_args.use_auth_token,
        data_args.is_preproc,
    )
    # Set seed before initializing model.
    set_seed(training_args.seed)
    tokenizer = create_tokenizer(model_args.model_name_or_path, model_args.vocab_file, model_args.cache_dir)
    model = create_model(
        model_args.model_name_or_path,
        len(tokenizer),
        model_args.config_name,
        model_args.cache_dir,
        model_args.use_auth_token,
    )

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    eval_dataset, predict_dataset, train_dataset = preprocess_dataset(tokenizer, data_args, datasets, training_args)

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    metric = get_metric(data_args.eval_metric)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = metric.run(decoded_preds, decoded_labels)

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_length=max_length,
            num_beams=num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    return results


def preprocess_dataset(tokenizer, data_args, datasets, training_args):
    """If the dataset was not processed already, do that now, otherwise return input dataset, possibly down-sampled"""
    train_dataset = None
    eval_dataset = None
    predict_dataset = None
    if training_args.do_train:
        train_dataset = datasets["train"]
        column_names = train_dataset.column_names
    elif training_args.do_eval:
        eval_dataset = datasets["validation"]
        column_names = eval_dataset.column_names
    elif training_args.do_predict:
        predict_dataset = datasets["test"].column_names
        column_names = predict_dataset.column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return
    logger.info(column_names)

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
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")

        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

        if not data_args.is_preproc:
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                train_dataset = preprocessor.run(train_dataset, desc="Running tokenizer on train dataset")
    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")

        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        if not data_args.is_preproc:
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                eval_dataset = preprocessor.run(eval_dataset, desc="Running tokenizer on evaluation dataset")
    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")

        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        if not data_args.is_preproc:
            with training_args.main_process_first(desc="prediction dataset map pre-processing"):
                predict_dataset = preprocessor.run(predict_dataset, desc="Running tokenizer on prediction dataset")
    return eval_dataset, predict_dataset, train_dataset


def last_available_checkpoint(training_args):
    """Detecting last checkpoint if it exists and we are not overwriting the output dir"""
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
