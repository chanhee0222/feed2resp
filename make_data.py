import argparse
import json
import os
import random
import re

from parlai.tasks.self_feeding.build import build
from tqdm import tqdm

from models.style_bart import BartSystem
from bart_utils import SummarizationDataset

person_sep_re = re.compile(r"__p[12]__")
person_num_re = re.compile(r"__p(?P<num>[12])__")


def download_self_feeding_data(args):
    # Download the data from paper "Learning from Dialogue after Deployment: Feed Yourself, Chatbot!"
    # https://www.aclweb.org/anthology/P19-1358.pdf
    build(vars(args))


def preprocess_data(args, context_last_n=2):
    # Preprocess the data as described in the paper "Learning Improvised Chatbots from Adversarial Modifications of Natural Language Feedback"
    # https://www.aclweb.org/anthology/2020.findings-emnlp.221.pdf

    self_feeding_root = os.path.join(args.datapath, "self_feeding")
    # Feedback. (jsonl format)
    feedback_file = os.path.join(self_feeding_root, "train_fb.txt")
    with open(feedback_file, "r") as in_fh:
        feedback_data = [json.loads(l.strip()) for l in in_fh]
    num_feedback = len(feedback_data)

    # Human-human dialogue.
    dialogue_file = os.path.join(self_feeding_root, "train_hh.txt")
    with open(dialogue_file, "r") as in_fh:
        dialogue_data = [json.loads(l.strip()) for l in in_fh]

    # Sample num_feedback number of dialogues.
    dialogue_data = random.sample(dialogue_data, num_feedback)

    # Preprocess a single row of data. Return a tuple (input, target).
    def make_example(row):
        context = row["context"].strip()
        feedback = row["response"].strip()

        context_utterances = person_sep_re.split(context)
        context_utterances = [u.strip() for u in context_utterances if len(u.strip()) > 0]
        seps = person_num_re.findall(context)

        assert len(seps) == len(context_utterances)

        # Select the last n utterances as context.
        if len(seps) > context_last_n:
            context_utterances = context_utterances[-context_last_n:]
            seps = seps[-context_last_n:]

        # Combine the context and feedback into a single example using separators.
        seps = [int(s) - 1 for s in seps]
        example = [f"{BartSystem.person_sep[sep]} {utt}" for sep, utt in zip(seps, context_utterances)]
        example.append(f"{BartSystem.response_sep} {feedback}")
        return " ".join(example), feedback

    # Preprocess the feedback data.
    feedback_examples = [make_example(row) for row in tqdm(feedback_data, desc="Loading feedbacks")]

    # Preprocess the dialogue data.
    dialogue_examples = [make_example(row) for row in tqdm(dialogue_data, desc="Loading dialogues")]

    # Shuffle.
    random.shuffle(feedback_examples)
    random.shuffle(dialogue_examples)

    def save_data(split, start=0, end=None):
        print(f"Writing '{split}' split.")
        with open(os.path.join(args.datapath, f"{split}.source"), "w") as src_fh, \
                open(os.path.join(args.datapath, f"{split}.target"), "w") as tgt_fh:
            # .source file has all the context + response + style label (0 = dialogue, 1 = feedback).
            # .target has only the response.
            for d_example, f_example in zip(dialogue_examples[start:end], feedback_examples[start:end]):
                # Alternate dialogue and feedback examples
                # so that each split has an equal number of examples for each class.
                d_src, d_tgt = d_example
                f_src, f_tgt = f_example

                src_fh.write(f"{d_src}{SummarizationDataset.label_sep}0\n")
                tgt_fh.write(f"{d_tgt}\n")

                src_fh.write(f"{f_src}{SummarizationDataset.label_sep}1\n")
                tgt_fh.write(f"{f_tgt}\n")

    # Train/valid/test split (8:1:1).
    num_valid = len(feedback_examples) // 10

    save_data("val", end=num_valid)
    save_data("test", start=num_valid, end=2*num_valid)
    save_data("train", start=2*num_valid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("datapath", help="Path to store the data.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(os.path.join(args.datapath, "args.json"), 'w') as out_fh:
        json.dump(vars(args), out_fh, indent=2, sort_keys=True)
    random.seed(args.seed)
    download_self_feeding_data(args)
    preprocess_data(args)
