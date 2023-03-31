import unittest
import torch
from alexa_teacher_models.tokenization_atm import AlexaTMTokenizerFast


def test_tokenizer():

    tokenizer = AlexaTMTokenizerFast(vocab_file="test-data/fake-spm.model")

    test = "this is fake data"
    encoded = tokenizer(test, return_tensors="pt")
    decoded = tokenizer.batch_decode(encoded['input_ids'], skip_special_tokens=True)
    expected_ids = [[1, 3, 9, 13, 7, 5, 3, 7, 5, 3, 12, 6, 8, 4, 3, 21, 6, 9, 6, 2]]
    assert encoded['input_ids'].tolist() == expected_ids
    assert decoded[0] == "<s> this is fake data</s>"
