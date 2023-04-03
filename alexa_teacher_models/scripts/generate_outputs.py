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

import sys
import os
import torch
from alexa_teacher_models import AlexaTMTokenizerFast
from alexa_teacher_models import AlexaTMSeq2SeqForConditionalGeneration
from datasets import load_dataset
from argparse import ArgumentParser
import nltk


def load_model_on_multiple_gpus(model_path, num_gpus=4):
    print('Loading the model...')
    tokenizer = AlexaTMTokenizerFast.from_pretrained(
        model_path, vocab_file=os.path.join(model_path, "20b_tokenizer.model")
    )
    model = AlexaTMSeq2SeqForConditionalGeneration.from_pretrained(model_path)

    model.bfloat16()
    if num_gpus == 1:
        model.to('cuda:0')
        return model, tokenizer, torch.device('cuda:0')
    elif num_gpus > 0:
        model.parallelize(num_gpus)
        return model, tokenizer, torch.device('cuda:0')
    else:
        return model, tokenizer, torch.device('cpu')


def download_data():
    dataset = dict()
    langs = {
        'ara': 'Arabic',
        'fra': 'French',
        'eng': 'English',
        'deu': 'German',
        'ita': 'Italian',
        'jpn': 'Japanese',
        'hin': 'Hindi',
        'mar': 'Marathi',
        'tam': 'Tamil',
        'tel': 'Telugu',
        'spa': 'Spanish',
    }

    for l in langs.keys():
        dataset[l] = load_dataset("gsarti/flores_101", l)

    return dataset, langs


def save_output(input_list, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w+') as f:
        for l in input_list:
            f.write(l + '\n')


def xsum(model, tokenizer, device, NSHOTS):
    dataset = load_dataset("xsum")

    # limit to pick short examples for shots
    shots_length_limit = 192

    short_examples = []
    for d in dataset['train']:
        if len(d['document'].split()) < shots_length_limit:
            short_examples.append(d)

    shots = ''
    for i in range(NSHOTS):
        shots += """Article: {0} ==> Short summary: {1} <br><br><br> """.format(
            short_examples[i]['document'], short_examples[i]['summary']
        )
    # shorten very long documents in the test
    src = []
    limit = 550
    for d in dataset['test']['document']:
        tokenized_doc = tokenizer.encode(d)
        doc_length = len(tokenized_doc)
        if doc_length > limit:
            test = tokenizer.decode(tokenized_doc[1:limit])
            test_sent = nltk.sent_tokenize(test)
            test = ' '.join(test_sent[:-1])
        else:
            test = d
        src.append("""[CLM] """ + shots + """Article: {0} ==> Short summary:""".format(test))
    # tgt = [s for s in dataset['test']['summary']]

    print('Generating summaries...')
    summaries = []
    for i in range(len(src)):
        test = src[i]
        if i == 0:
            print('Example input to the model: \n', test)
        encoded = tokenizer(test, return_tensors="pt").to(device)
        with torch.no_grad():
            generated_tokens = model.generate(
                input_ids=encoded['input_ids'],
                max_length=64,
                num_beams=1,
                num_return_sequences=1,
                bad_words_ids=[[0], [2012, 2006]],
                early_stopping=True,
            )
        summaries.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])

    cleaned_summaries = []
    for i in range(len(summaries)):
        tmp = summaries[i]
        e = tmp.find('<br><br><br>')
        if e > 0:
            cleaned_summaries.append(tmp[:e])
        else:
            cleaned_summaries.append(tmp[: tmp.find('</s>')])

    save_output(cleaned_summaries, os.path.join(os.getcwd(), 'xsum/summaries' + '_20B_{}shot.txt'.format(NSHOTS)))


def flores_101(model, tokenizer, device, NSHOTS):

    # download data
    dataset = dict()
    langs = {
        'ara': 'Arabic',
        'fra': 'French',
        'eng': 'English',
        'deu': 'German',
        'ita': 'Italian',
        'jpn': 'Japanese',
        'hin': 'Hindi',
        'mar': 'Marathi',
        'tam': 'Tamil',
        'tel': 'Telugu',
        'spa': 'Spanish',
    }

    for l in langs.keys():
        dataset[l] = load_dataset("gsarti/flores_101", l)

    for ls in langs.keys():
        for lt in langs.keys():
            if ls == lt:
                continue
            print('Generating outputs for flores_101 devtest set for ' + ls + ' --> ', lt)
            shots = ''
            for i in range(NSHOTS):
                shots += """Sentence:  {0}; Translation in {1}: {2}; """.format(
                    dataset[ls]['dev'][i]['sentence'], langs[lt], dataset[lt]['dev'][i]['sentence']
                )

            translated = []
            tgt = []
            for i, d in enumerate(dataset[ls]['devtest']):

                test = (
                    """[CLM] """ + shots + """Sentence:  {0}; Translation in {1}:""".format(d['sentence'], langs[lt])
                )
                if i == 0:
                    print('Example input to the model: \n', test)
                tgt.append(dataset[lt]['devtest'][i]['sentence'])
                encoded = tokenizer(test, return_tensors="pt").to(device)
                with torch.no_grad():
                    generated_tokens = model.generate(
                        input_ids=encoded['input_ids'],
                        max_length=64,
                        num_beams=1,
                        num_return_sequences=1,
                        bad_words_ids=[[0], [2012, 2006]],
                        early_stopping=True,
                    )
                translated.append(tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0])

            cleaned_translated = []
            for i in range(len(translated)):
                tmp = translated[i]
                if tmp.find('Translation') > 0:
                    ind = tmp.find('Translation') - 1
                else:
                    ind = len(tmp)
                clean = tmp[: min(tmp.find(';') % len(tmp), ind)]
                cleaned_translated.append(clean)

            save_output(
                cleaned_translated,
                os.path.join(os.getcwd(), 'flores101_outputs/' + ls + 'T' + lt + '_20B_{}shot.txt'.format(NSHOTS)),
            )


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("-p", "--model-path", required=True, type=str)
    arg_parser.add_argument("-n", "--number-of-shots", default=0, type=int)
    arg_parser.add_argument("-g", "--number-of-gpus", default=0, type=int)
    arg_parser.add_argument("-t", "--task", required=True, type=str, help='supported tasks: [flores_101, xsum]')
    args = arg_parser.parse_args(sys.argv[1:])

    assert (args.number_of_gpus % 2 == 0 or args.number_of_gpus == 1) and args.number_of_gpus <= 8, (
        "You can either use 1 GPU or " "an even number of GPUs up to 8"
    )

    model, tokenizer, device = load_model_on_multiple_gpus(args.model_path, num_gpus=args.number_of_gpus)
    print(f'Generating outputs for {args.task} task using AlexaTM 20B using {args.number_of_shots} shots!')
    if args.task == 'flores_101':
        flores_101(model, tokenizer, device, args.number_of_shots)
    if args.task == 'xsum':
        xsum(model, tokenizer, device, args.number_of_shots)
    else:
        print(f'Task {args.task} is not supported!')


if __name__ == "__main__":
    main()
