import csv
import json
import os

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

SPECIAL_TOKENS = {"[SEP]", "[CLS]", "[PAD]"}


class DialoglueIntentDataset(Dataset):
    def __init__(self, data_path: str, intent_label_to_idx=None):

        data_dirname = os.path.dirname(os.path.abspath(data_path))

        # Intent categories
        intent_vocab_path = os.path.join(data_dirname, "categories.json")
        intent_names = json.load(open(intent_vocab_path))
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
        with open(data_path) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)

            for utt, intent in tqdm(reader):
                self.examples.append(
                    {"Text": utt, "Label": self.intent_label_to_idx[intent]}
                )

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
        df = pd.read_csv(data_path)
        intent_names = list(set(df["label"])) + ["NO_NODES_DETECTED"]
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

        for row in df.itertuples(index=False):
            self.examples.append(
                {"Text": row.sentence, "Label": self.intent_label_to_idx[row.label]}
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
