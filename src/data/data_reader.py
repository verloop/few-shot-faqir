import chunk
import csv
import json
import os

import pandas as pd
from sentence_transformers import InputExample
from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm

SPECIAL_TOKENS = {"[SEP]", "[CLS]", "[PAD]"}


class DialoglueIntentDataset(Dataset):
    def __init__(self, data_path: str, intent_label_to_idx=None):

        data_dirname = os.path.dirname(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        intent_names = json.load(open(intent_vocab_path))
        intent_names = intent_names + ["oos"]
        if intent_label_to_idx:
            self.intent_label_to_idx = intent_label_to_idx
        else:
            self.intent_label_to_idx = dict(
                (label, idx) for idx, label in enumerate(intent_names)
            )
        self.intent_idx_to_label = {
            idx: label for label, idx in self.intent_label_to_idx.items()
        }

        # Process data
        self.examples = []
        self.df = pd.read_csv(data_path)

        for row in self.df.itertuples(index=False):
            self.examples.append(
                {"Text": row.text, "Label": self.intent_label_to_idx[row.category]}
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DialoglueSbertIntentDataset(Dataset):
    def __init__(self, data_path: str, intent_label_to_idx=None):

        data_dirname = os.path.dirname(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        intent_names = json.load(open(intent_vocab_path))
        intent_names = intent_names + ["oos"]
        if intent_label_to_idx:
            self.intent_label_to_idx = intent_label_to_idx
        else:
            self.intent_label_to_idx = dict(
                (label, idx) for idx, label in enumerate(intent_names)
            )
        self.intent_idx_to_label = {
            idx: label for label, idx in self.intent_label_to_idx.items()
        }

        # Process data
        self.examples = []
        self.df = pd.read_csv(data_path)

        for row in self.df.itertuples(index=False):
            input_example = InputExample(
                texts=[row.text], label=float(self.intent_label_to_idx[row.category])
            )
            self.examples.append(input_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class DialoglueTOPDataset(Dataset):
    def __init__(self, data_path: str, intent_label_to_idx=None):
        data_dirname = os.path.dirname(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "vocab.intent")
        intent_names = [e.strip() for e in open(intent_vocab_path).readlines()]
        if intent_label_to_idx:
            self.intent_label_to_idx = intent_label_to_idx
        else:
            self.intent_label_to_idx = dict(
                (label, idx) for idx, label in enumerate(intent_names)
            )
        self.intent_idx_to_label = {
            idx: label for label, idx in self.intent_label_to_idx.items()
        }

        # Process data
        self.examples = []
        data = []
        with open(data_path) as csvfile:
            data = [ex.strip() for ex in csvfile.readlines()]

        for example in tqdm(data):
            example, intent = example.split(" <=> ")
            text = " ".join([e.split(":")[0] for e in example.split()])

            self.examples.append(
                {"Text": text, "Label": self.intent_label_to_idx[intent]}
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class HapticDataset(Dataset):
    def __init__(self, data_path: str, intent_label_to_idx=None):
        self.df = pd.read_csv(data_path)
        intent_names = list(set(self.df["label"])) + ["NO_NODES_DETECTED"]
        if intent_label_to_idx:
            self.intent_label_to_idx = intent_label_to_idx
        else:
            self.intent_label_to_idx = dict(
                (label, idx) for idx, label in enumerate(intent_names)
            )

        self.intent_idx_to_label = {
            idx: label for label, idx in self.intent_label_to_idx.items()
        }

        self.examples = []

        for row in self.df.itertuples(index=False):
            self.examples.append(
                {"Text": row.sentence, "Label": self.intent_label_to_idx[row.label]}
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class HapticSbertDataset(Dataset):
    def __init__(self, data_path: str, intent_label_to_idx=None):
        self.df = pd.read_csv(data_path)
        intent_names = list(set(self.df["label"])) + ["NO_NODES_DETECTED"]
        if intent_label_to_idx:
            self.intent_label_to_idx = intent_label_to_idx
        else:
            self.intent_label_to_idx = dict(
                (label, idx) for idx, label in enumerate(intent_names)
            )

        self.intent_idx_to_label = {
            idx: label for label, idx in self.intent_label_to_idx.items()
        }

        self.examples = []

        for row in self.df.itertuples(index=False):
            inp_example = InputExample(
                texts=[row.sentence], label=float(self.intent_label_to_idx[row.label])
            )
            self.examples.append(inp_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class QuestionPairDataset(Dataset):
    def __init__(self, data_path: str):
        df = pd.read_csv(data_path)
        if len(df) < 200000:
            df = df.sample(n=200000, replace=True, random_state=42)
        else:
            df = df.sample(n=200000, random_state=42)
        self.examples = []

        for row in df.itertuples(index=False):
            self.examples.append((row.question1, row.question2, row.label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class QuestionPairTestTrainDataset(Dataset):
    """
    Gets question pairs of test and train combinations along with test indx and the label - 1/0
    If the train label and test matches , its 1 which will only be for the questions belonging to the true label
    """

    def __init__(self, data_path: str):
        df = pd.read_csv(data_path)
        self.examples = []

        for row in df.itertuples(index=False):
            self.examples.append((row.question1, row.question2, row.label, row.idx))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class QuestionPairSentBertDataset(Dataset):
    def __init__(self, data_path: str):
        df = pd.read_csv(data_path)
        if len(df) < 200000:
            df = df.sample(n=200000, replace=True, random_state=42)
        else:
            df = df.sample(
                n=200000, random_state=42
            )  # Addded for Sentence Bert cross encoder training which doesnt have STEPS_PER_EPOCH
        self.examples = []

        for row in df.itertuples(index=False):
            inp_example = InputExample(
                texts=[row.question1, row.question2], label=float(row.label)
            )
            self.examples.append(inp_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class QuestionTripletsSentBertDataset(Dataset):
    def __init__(self, data_path: str):
        df = pd.read_csv(data_path)
        if len(df) < 200000:
            df = df.sample(n=200000, replace=True, random_state=42)
        else:
            df = df.sample(
                n=200000, random_state=42
            )  # Addded for Sentence Bert cross encoder training which doesnt have STEPS_PER_EPOCH
        self.examples = []

        for row in df.itertuples(index=False):
            inp_example = InputExample(texts=[row.anchor, row.positive, row.negative])
            self.examples.append(inp_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class QuestionPairChunkedSentBertDataset(Dataset):
    def __init__(self, data_path: str):
        df_iter = pd.read_csv(data_path, chunksize=500)
        self.examples = []
        for df in df_iter:
            for row in df.itertuples(index=False):
                inp_example = InputExample(
                    texts=[row.question1, row.question2], label=float(row.label)
                )
                self.examples.append(inp_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class QuestionTripletsChunkedSentBertDataset(Dataset):
    def __init__(self, data_path: str):
        df_iter = pd.read_csv(data_path, chunksize=500)
        self.examples = []
        for df in df_iter:
            for row in df.itertuples(index=False):
                inp_example = InputExample(
                    texts=[row.anchor, row.positive, row.negative]
                )
                self.examples.append(inp_example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class QuestionPairIterableSentBertDataset(IterableDataset):
    def __init__(self, data_path: str):
        self.filename = data_path

    def __getitem__(self, idx):
        return self.examples[idx]

    def line_mapper(self, line):
        line = line.replace("\n", "")
        question1, question2, label = line.split(",")
        inp_example = InputExample(texts=[question1, question2], label=float(label))
        return inp_example

    def __iter__(self):
        file_itr = open(self.filename)
        mapped_itr = map(self.line_mapper, file_itr)
        return mapped_itr
