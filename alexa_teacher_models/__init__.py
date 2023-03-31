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
