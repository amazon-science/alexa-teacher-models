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

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# This import makes AlexaTM available as auto models
import alexa_teacher_models
import os
import logging
from pprint import pprint
from prompt_toolkit import PromptSession
from alexa_teacher_models.scripts.train_utils import create_tokenizer

DEFAULT_PRE = '[CLM] '
# TODO: make configurable level
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


def run(prompt, tokenizer, model, max_length, device, do_sample):
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        ids = model.generate(
            input_ids=inputs['input_ids'],
            max_length=max_length,
            do_sample=do_sample,
            num_return_sequences=1,
            bad_words_ids=[[0], [2012, 2006]],
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--pre", type=str, help="prefix", default=DEFAULT_PRE)
    parser.add_argument(
        "--max_history", type=int, default=0, help="How much contextual history to keep from the session"
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=40)
    parser.add_argument("--join_string", type=str, default=' ')
    parser.add_argument("--vocab_file", type=str)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", type=str)
    parser.add_argument("--do_sample", action="store_true")

    args = parser.parse_args()
    tokenizer = create_tokenizer(args.model, args.vocab_file)
    logger.info("Created tokenizer")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    logger.info("Created model")

    available_devices = torch.cuda.device_count()
    if not args.device:
        args.device = 'cpu' if not available_devices else 'cuda:0'

    model.bfloat16()
    logger.info("Placing model on device (%s)", args.device)

    if args.device != 'cpu':
        logger.info("Parallelizing the model across [%d] devices", available_devices)
        if available_devices % 2 == 0:
            model.parallelize(available_devices)
        else:
            model.to('cuda:0')

    convo = []
    session = PromptSession()
    print('Session')
    print('=' * 40)
    print(args.pre)
    while True:
        try:
            from_user = session.prompt("=> ")

            if from_user.startswith(":q"):
                break
            if from_user.startswith(":cl"):
                print('clearing session')
                convo = []
                continue
            convo.append(from_user)
            prompt = args.pre + args.join_string.join(convo[-(args.max_history + 1) :])
            if args.verbose:
                print(prompt)
            text = run(prompt, tokenizer, model, args.max_length, args.device, args.do_sample)
            pprint(text)
            convo.append(text)
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
    print("bye")


if __name__ == '__main__':
    main()
