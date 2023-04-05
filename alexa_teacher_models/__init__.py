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
__version__ = "1.0.1"

from .configuration_atm import (
    AlexaTMConfig,
    AlexaTMSeq2SeqConfig,
)
from .tokenization_atm import AlexaTMTokenizerFast, register_auto_tokenizer
from .modeling_atm import (
    AlexaTM,
    AlexaTMForMaskedLM,
    AlexaTMSeq2SeqModel,
    AlexaTMSeq2SeqForConditionalGeneration,
    register_auto_model,
)

from transformers.utils import logging

logger = logging.get_logger(__name__)

try:

    register_auto_model()
    logger.info("Successfully registered AlexaTM model as AutoModel")
    try:
        register_auto_tokenizer()
        logger.info("Successfully registered AlexaTM fast tokenizer as AutoTokenizer")
    except Exception as e:
        logger.warning("Failed to register AutoTokenizer")
        logger.error(e)
except Exception as e:
    logger.warning("Failed to register AutoModel")
    logger.error(e)
