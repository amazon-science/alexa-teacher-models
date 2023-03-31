import unittest
import torch
from alexa_teacher_models.modeling_atm import (
    AlexaTMSeq2SeqConfig,
    AlexaTMSeq2SeqModel,
    AlexaTMSeq2SeqForConditionalGeneration,
)


def test_seq2seq():
    config = AlexaTMSeq2SeqConfig(
        add_bias_logits=False,
        add_final_layer_norm=False,
        bos_token_id=1,
        eos_token_id=2,
        d_model=128,
        decoder_attention_heads=16,
        decoder_ffn_dim=128,
        decoder_layers=2,
        decoder_start_token_id=1,
        do_blenderbot_90_layernorm=False,
        encoder_attention_heads=16,
        encoder_ffn_dim=128,
        encoder_layers=2,
        extra_pos_embeddings=2,
        force_bos_token_to_be_generated=False,
        forced_eos_token_id=2,
        is_encoder_decoder=True,
        max_position_embeddings=10,
        share_embeddings=True,
        type_vocab_size=1,
        vocab_size=10,
    )
    model = AlexaTMSeq2SeqModel(config)

    fake_data = torch.tensor([[1, 2, 3, 4]])
    attn_mask = torch.ones_like(fake_data)

    output = model(fake_data, attn_mask)
    assert output.last_hidden_state.shape[0] == 1
    assert output.last_hidden_state.shape[1] == 4
    assert output.last_hidden_state.shape[2] == 128

    assert output.encoder_last_hidden_state.shape[0] == 1
    assert output.encoder_last_hidden_state.shape[1] == 4
    assert output.encoder_last_hidden_state.shape[2] == 128


def test_seq2seq_generate():
    config = AlexaTMSeq2SeqConfig(
        add_bias_logits=False,
        add_final_layer_norm=False,
        bos_token_id=1,
        eos_token_id=2,
        d_model=128,
        decoder_attention_heads=16,
        decoder_ffn_dim=128,
        decoder_layers=2,
        decoder_start_token_id=1,
        do_blenderbot_90_layernorm=False,
        encoder_attention_heads=16,
        encoder_ffn_dim=128,
        encoder_layers=2,
        extra_pos_embeddings=2,
        force_bos_token_to_be_generated=False,
        forced_eos_token_id=2,
        is_encoder_decoder=True,
        max_position_embeddings=10,
        share_embeddings=True,
        type_vocab_size=1,
        vocab_size=10,
    )
    model = AlexaTMSeq2SeqForConditionalGeneration(config)

    fake_data = torch.tensor([[1, 2, 3, 4]])

    output = model.generate(
        fake_data,
        min_length=6,
        max_length=6,
        num_beams=1,
    )

    assert list(output.shape) == [1, 6]
