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

import datasets
import torch
import os
import logging
import warnings
import sys
import alexa_teacher_models
from transformers.utils import logging as transformers_logging
from transformers.trainer_utils import (
    EvaluationStrategy,
    HubStrategy,
    IntervalStrategy,
    SchedulerType,
)


from transformers.training_args import (
    OptimizerNames,
    default_logdir,
    get_xla_device_type,
    is_torch_available,
    is_torch_tf32_available,
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_tpu_available,
)
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments as Seq2SeqTrainingBase

logger = logging.getLogger(__name__)


def create_tokenizer(model_name_or_path, vocab_file=None, cache_dir=None):
    """Some models use a different vocab filename instead of the defaults.  Set it in factory method if provided"""
    if vocab_file is not None:
        return AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir, vocab_file=vocab_file)
    else:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
        )


def load_data(
    train_file=None,
    validation_file=None,
    test_file=None,
    dataset_name=None,
    dataset_config_name=None,
    cache_dir=None,
    use_auth_token=False,
    is_preproc=False,
):
    """Load dataset by name, by local JSON or CSV files, or from a preprocessed written using ``save_to_disk()``"""

    if is_preproc:
        proc_data = {}
        if train_file is not None:
            proc_data["train"] = datasets.load_from_disk(train_file)
        if validation_file is not None:
            proc_data["validation"] = datasets.load_from_disk(validation_file)
        if test_file is not None:
            proc_data["test"] = datasets.load_from_disk(test_file)
        return proc_data

    if dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = datasets.load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
            use_auth_token=True if use_auth_token else None,
        )
    else:
        data_files = {}
        if train_file is not None:
            data_files["train"] = train_file
            extension = train_file.split(".")[-1]
        if validation_file is not None:
            data_files["validation"] = validation_file
            extension = validation_file.split(".")[-1]
        if test_file is not None:
            data_files["test"] = test_file
            extension = test_file.split(".")[-1]
        raw_datasets = datasets.load_dataset(
            extension,
            data_files=data_files,
            cache_dir=cache_dir,
            use_auth_token=True if use_auth_token else None,
        )
    return raw_datasets


class VoltaBFloat16Seq2SeqTrainingArguments(Seq2SeqTrainingBase):
    def __post_init__(self):
        # Handle --use_env option in torch.distributed.launch (local_rank not passed as an arg then).
        # This needs to happen before any call to self.device or self.n_gpu.
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1 and env_local_rank != self.local_rank:
            self.local_rank = env_local_rank

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        # see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        if isinstance(self.evaluation_strategy, EvaluationStrategy):
            warnings.warn(
                "using `EvaluationStrategy` for `evaluation_strategy` is deprecated and will be removed in version 5"
                " of 🤗 Transformers. Use `IntervalStrategy` instead",
                FutureWarning,
            )
            # Go back to the underlying string or we won't be able to instantiate `IntervalStrategy` on it.
            self.evaluation_strategy = self.evaluation_strategy.value

        self.evaluation_strategy = IntervalStrategy(self.evaluation_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = IntervalStrategy(self.save_strategy)
        self.hub_strategy = HubStrategy(self.hub_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.evaluation_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.evaluation_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        # Sanity checks for load_best_model_at_end: we require save and eval strategies to be compatible.
        if self.load_best_model_at_end:
            if self.evaluation_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval strategy to match, but found\n- Evaluation "
                    f"strategy: {self.evaluation_strategy}\n- Save strategy: {self.save_strategy}"
                )
            if self.evaluation_strategy == IntervalStrategy.STEPS and self.save_steps % self.eval_steps != 0:
                raise ValueError(
                    "--load_best_model_at_end requires the saving steps to be a round multiple of the evaluation "
                    f"steps, but found {self.save_steps}, which is not a round multiple of {self.eval_steps}."
                )

        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.metric_for_best_model not in ["loss", "eval_loss"]
        if self.run_name is None:
            self.run_name = self.output_dir
        if self.framework == "pt" and is_torch_available():
            if self.fp16_backend and self.fp16_backend != "auto":
                warnings.warn(
                    "`fp16_backend` is deprecated and will be removed in version 5 of 🤗 Transformers. Use"
                    " `half_precision_backend` instead",
                    FutureWarning,
                )
                self.half_precision_backend = self.fp16_backend

            if self.bf16 or self.bf16_full_eval:

                if self.no_cuda and not is_torch_bf16_cpu_available() and not is_torch_tpu_available():
                    # cpu
                    raise ValueError("Your setup doesn't support bf16/(cpu, tpu, neuroncore). You need torch>=1.10")
                elif not self.no_cuda and torch.cuda.is_available() and not is_torch_bf16_gpu_available():
                    # gpu
                    warnings.warn(
                        "Your setup doesn't support bf16/gpu so training will be slow. For intrinsics you need torch>=1.10, using Ampere GPU with cuda>=11.0"
                    )

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")

        if self.fp16_full_eval and self.bf16_full_eval:
            raise ValueError("At most one of fp16 and bf16 can be True for full eval, but not both")

        if self.bf16:
            if self.half_precision_backend == "apex":
                raise ValueError(
                    " `--half_precision_backend apex`: GPU bf16 is not supported by apex. Use"
                    " `--half_precision_backend cuda_amp` instead"
                )

        self.optim = OptimizerNames(self.optim)

        if (
            self.framework == "pt"
            and is_torch_available()
            and (self.device.type != "cuda")
            and (get_xla_device_type(self.device) != "GPU")
            and (self.fp16 or self.fp16_full_eval)
        ):
            raise ValueError(
                "FP16 Mixed precision training with AMP or APEX (`--fp16`) and FP16 half precision evaluation"
                " (`--fp16_full_eval`) can only be used on CUDA devices."
            )

        if (
            self.framework == "pt"
            and is_torch_available()
            and (self.device.type != "cuda")
            and (get_xla_device_type(self.device) != "GPU")
            and (get_xla_device_type(self.device) != "TPU")
            and (self.device.type != "cpu")
            and (self.bf16 or self.bf16_full_eval)
        ):
            raise ValueError(
                "BF16 Mixed precision training with AMP (`--bf16`) and BF16 half precision evaluation"
                " (`--bf16_full_eval`) can only be used on CUDA or CPU/TPU/NeuronCore devices."
            )

        if (self.torch_compile_mode is not None or self.torch_compile_backend is not None) and not self.torch_compile:
            self.torch_compile = True
        if self.torch_compile and self.torch_compile_backend is None:
            self.torch_compile_backend = "inductor"
        if self.framework == "pt" and is_torch_available() and self.torch_compile:
            if is_torch_tf32_available():
                if self.tf32 is None and not self.fp16 or self.bf16:
                    logger.info(
                        "Setting TF32 in CUDA backends to speedup torch compile, you won't see any improvement"
                        " otherwise."
                    )
                    torch.backends.cuda.matmul.allow_tf32 = True
            else:
                logger.warning(
                    "The speedups for torchdynamo mostly come wih GPU Ampere or higher and which is not detected here."
                )
        if self.framework == "pt" and is_torch_available() and self.tf32 is not None:
            if self.tf32:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = True
                else:
                    raise ValueError("--tf32 requires Ampere or a newer GPU arch, cuda>=11 and torch>=1.7")
            else:
                if is_torch_tf32_available():
                    torch.backends.cuda.matmul.allow_tf32 = False
                # no need to assert on else

        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from transformers.integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio"
                " during training"
            )

        if self.deepspeed:
            # - must be run very last in arg parsing, since it will use a lot of these settings.
            # - must be run before the model is created.
            from transformers.deepspeed import HfTrainerDeepSpeedConfig

            # will be used later by the Trainer
            # note: leave self.deepspeed unmodified in case a user relies on it not to be modified)
            self.hf_deepspeed_config = HfTrainerDeepSpeedConfig(self.deepspeed)
            self.hf_deepspeed_config.trainer_config_process(self)


class Preprocessor:
    def __init__(
        self,
        tokenizer,
        column_names,
        max_length=1024,
        max_target_length=128,
        source_field='x',
        target_field='y',
        prefix='',
        padding=True,
        ignore_pad_token_for_loss=True,
        num_workers=4,
        load_from_cache_file=True,
    ):
        self.tokenizer = tokenizer
        self.column_names = column_names
        self.max_length = max_length
        self.max_target_length = max_target_length
        self.source_field = source_field
        self.target_field = target_field
        self.prefix = prefix
        self.padding = padding
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.num_workers = num_workers
        self.allow_load_from_cache = load_from_cache_file

    def run(self, dataset, desc=""):
        def preprocess_function(examples):
            inputs = []
            targets = []

            for i, o in zip(examples[self.source_field], examples[self.target_field]):
                if not i or not o:
                    logger.info("Skipping empty source or dest")
                    continue
                inputs.append(i)
                targets.append(o)

            inputs = [self.prefix + inp for inp in inputs]
            model_inputs = self.tokenizer(inputs, max_length=self.max_length, padding=self.padding, truncation=True)
            labels = self.tokenizer(
                text_target=targets,
                max_length=self.max_target_length,
                padding=self.padding,
                truncation=True,
            )

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if self.padding == "max_length" and self.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(label_token if label_token != self.tokenizer.pad_token_id else -100) for label_token in label]
                    for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        return dataset.map(
            preprocess_function,
            batched=True,
            num_proc=self.num_workers,
            remove_columns=self.column_names,
            load_from_cache_file=self.allow_load_from_cache,
            desc=desc,
        )


def setup_logging(logger, log_level):
    """Setup logging"""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers_logging.set_verbosity(log_level)
    transformers_logging.enable_default_handler()
    transformers_logging.enable_explicit_format()
