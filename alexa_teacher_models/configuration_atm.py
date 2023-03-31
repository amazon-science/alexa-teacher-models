# Original Copyright 2021 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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

# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

# This is mainly copied from transformers.modeling_bart
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class AlexaTMConfig(RobertaConfig):
    """For use with AlexaTMPreTrainedModel.

    Args:

    """

    model_type = "preln"

    def __init__(self, **kwargs):
        """Constructs AlexaTMConfig."""
        super().__init__(**kwargs)


class AlexaTMSeq2SeqConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a
    :class:`alexa_teacher_models.models.alexatm_seq2seq`. It is used to
    instantiate a AlexaTM 20B seq2seq model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 50265):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~AlexaTMSeq2Seq`.
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed to encoder when calling :class:`~AlexaTMSeq2Seq`
        d_model (:obj:`int`, `optional`, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        encoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of encoder layers, 6 are used for the `bart-base` model.
        decoder_layers (:obj:`int`, `optional`, defaults to 12):
            Number of decoder layers, 6 are used for the `bart-base` model.
        encoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        decoder_attention_heads (:obj:`int`, `optional`, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        decoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        encoder_ffn_dim (:obj:`int`, `optional`, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        hidden_act (:obj:`str` or :obj:`function`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and decoder. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        dropout (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        activation_dropout (:obj:`float`, `optional`, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 1024):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        init_std (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        eos_token_id (:obj:`int`, `optional`, defaults to 2)
            End of stream token id.
        pad_token_id (:obj:`int`, `optional`, defaults to 1)
            Padding token id.
        bos_token_id (:obj:`int`, `optional`, defaults to 0)
            Beginning of stream token id.
        encoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the encoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        decoder_layerdrop: (:obj:`float`, `optional`, defaults to 0.0):
            The LayerDrop probability for the decoder. See the `LayerDrop paper <see
            https://arxiv.org/abs/1909.11556>`__ for more details.
        extra_pos_embeddings: (:obj:`int`, `optional`, defaults to 2):
            How many extra learned positional embeddings to use. Should be set to :obj:`pad_token_id+1`.
        is_encoder_decoder (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this is an encoder/decoder model.
        force_bos_token_to_be_generated (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to force BOS token to be generated at step 1 (after ``decoder_start_token_id``), only
            :obj:`True` for `bart-large-cnn`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        gradient_checkpointing (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If True, use gradient checkpointing to save memory at the expense of slower backward pass.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding in encoder. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        share_embeddings (:obj:`bool`, `optional`, defaults to :obj:`true`):
            Whether or not share embeddings between encoder and decoder
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in encoder.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities in encoder.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices in encoder.
    """
    model_type = "alexatm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50265,
        type_vocab_size=2,
        d_model=1024,
        encoder_layers=12,
        decoder_layers=12,
        encoder_attention_heads=16,
        decoder_attention_heads=16,
        decoder_ffn_dim=4096,
        encoder_ffn_dim=4096,
        hidden_act="gelu",
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        max_position_embeddings=1024,
        init_std=0.02,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        extra_pos_embeddings=2,
        is_encoder_decoder=True,
        force_bos_token_to_be_generated=False,
        use_cache=True,
        gradient_checkpointing=False,
        position_embedding_type="absolute",
        share_embeddings=True,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        fp32_cast_layer_norm=False,
        fp32_cast_query_key=None,
        **common_kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **common_kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        # pos embedding offset
        self.extra_pos_embeddings = extra_pos_embeddings

        self.d_model = d_model  # encoder_embed_dim and decoder_embed_dim
        self.encoder_layers = self.num_hidden_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.intermediate_size = encoder_ffn_dim
        self.hidden_act = hidden_act

        self.init_std = init_std  # Normal(0, this parameter)
        self.layer_norm_eps = layer_norm_eps

        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        self.use_cache = use_cache
        self.force_bos_token_to_be_generated = force_bos_token_to_be_generated
        self.gradient_checkpointing = gradient_checkpointing

        # This is unique to AlexaTM since it does not share the embeddings between encoder and decoder
        self.share_embeddings = share_embeddings

        # encoder related parameters (from BERTConfig)
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.position_embedding_type = position_embedding_type

        self.fp32_cast_layer_norm = fp32_cast_layer_norm
        self.fp32_cast_query_key = fp32_cast_query_key

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model
