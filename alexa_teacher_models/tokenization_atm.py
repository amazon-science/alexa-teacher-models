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

import os
from shutil import copyfile
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast, PreTrainedTokenizerBase
from transformers.convert_slow_tokenizer import SpmConverter
from tokenizers import processors
import sentencepiece as spm
from transformers.utils import logging

VOCAB_FILES_NAMES = {"vocab_file": "20b_tokenizer.model", "tokenizer_file": "tokenizer.json"}


logger = logging.get_logger(__name__)


class AlexaTMConverter(SpmConverter):
    def __init__(self, *args, create_type_ids=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_type_ids = create_type_ids

    def post_processor(self):
        if self.create_type_ids:
            return processors.TemplateProcessing(
                single=["<s>:0", "$A:0", "</s>:0"],
                pair=["<s>:0", "$A:0", "</s>:0", "$B:1", "</s>:1"],
                special_tokens=[
                    ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                    ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
                ],
            )
        else:
            return processors.TemplateProcessing(
                single=["<s>", "$A", "</s>"],
                pair=["<s>", "$A", "</s>", "$B", "</s>"],
                special_tokens=[
                    ("<s>", self.original_tokenizer.convert_tokens_to_ids("<s>")),
                    ("</s>", self.original_tokenizer.convert_tokens_to_ids("</s>")),
                ],
            )


class AlexaTMTokenizerFast(PreTrainedTokenizerFast):

    vocab_files_names = VOCAB_FILES_NAMES

    class AlexaTMTokenizer(PreTrainedTokenizer):
        def __init__(
            self,
            vocab_file,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            sep_token="</s>",
            pad_token="[PAD]",
            cls_token="<s>",
            mask_token="[MASK]",
            **kwargs,
        ):
            super().__init__(
                bos_token=bos_token,
                eos_token=eos_token,
                unk_token=unk_token,
                sep_token=sep_token,
                pad_token=pad_token,
                cls_token=cls_token,
                mask_token=mask_token,
                **kwargs,
            )
            self.vocab_file = vocab_file
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(self.vocab_file)

        def convert_tokens_to_ids(self, token):
            return self.sp_model.piece_to_id(token)

    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        sep_token="</s>",
        pad_token="[PAD]",
        cls_token="<s>",
        mask_token="[MASK]",
        create_segment_ids=False,
        **kwargs,
    ):
        self.vocab_file = vocab_file
        self.create_segment_ids = create_segment_ids
        slow_tokenizer = AlexaTMTokenizerFast.AlexaTMTokenizer(vocab_file, **kwargs)
        self._tokenizer = self.create_fast_tokenizer(slow_tokenizer, create_segment_ids)
        PreTrainedTokenizerBase.__init__(
            self,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        logger.info(
            "Initialized AlexaTMTokenizerFast using the following vocab: {} "
            "and special tokens: mask_token:{}, pad_token:{}, bos_token:{}, eos_token:{}, "
            "unk_token:{}, sep_token:{}, cls_token:{}, create_segment_ids:{}".format(
                vocab_file,
                mask_token,
                pad_token,
                bos_token,
                eos_token,
                unk_token,
                sep_token,
                cls_token,
                create_segment_ids,
            )
        )

    @staticmethod
    def create_fast_tokenizer(slow_tokenizer, create_token_type_id):
        return AlexaTMConverter(slow_tokenizer, create_type_ids=create_token_type_id).converted()

    def save_vocabulary(self, save_directory, filename_prefix):
        if not os.path.isdir(save_directory):
            logger.error(f"ERROR: Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
            logger.info(f"Copy vocab file to {out_vocab_file}")

        return (out_vocab_file,)


def register_auto_tokenizer():
    from transformers import AutoTokenizer

    AutoTokenizer.register("alexatm", None, AlexaTMTokenizerFast)
