import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.data_reader import (  # isort:skip
    DialoglueIntentDataset,
    DialoglueTOPDataset,
    HapticDataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_batch(batch, tokenizer: AutoTokenizer = None):
    label_list, text_list = [], []

    for example in batch:
        label_list.append(example["Label"])
        text_list.append(example["Text"])
    print(text_list[0])
    print(label_list[0])
    if tokenizer:
        batch_output = tokenizer(
            text_list, padding=True, truncation=True, return_tensors="pt"
        )
    else:
        batch_output = torch.tensor(text_list, dtype=torch.int64)

    label_list = torch.tensor(label_list, dtype=torch.int64)

    return label_list, batch_output


def collate_batch(batch, tokenizer: AutoTokenizer = None):
    label_list, text_list = tokenize_batch(batch, tokenizer)
    return text_list.to(device), label_list.to(device)


class HaptikDataLoader:
    def __init__(
        self, data_path="data/haptik/train/curekart_train.csv", intent_label_to_idx=None
    ):
        self.data_path = data_path
        print(f"Loading data from {self.data_path}")
        self.dataset = HapticDataset(self.data_path, intent_label_to_idx)

    def get_dataloader(
        self, batch_size=4, shuffle=True, tokenizer: AutoTokenizer = None
    ):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tokenizer),
        )


class DialogueIntentDataLoader:
    def __init__(
        self, data_path="data/dialoglue/banking/train.csv", intent_label_to_idx=None
    ):
        self.data_path = data_path
        print(f"Loading data from {self.data_path}")
        self.dataset = DialoglueIntentDataset(self.data_path, intent_label_to_idx)

    def get_dataloader(
        self, batch_size=4, shuffle=True, tokenizer: AutoTokenizer = None
    ):

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tokenizer),
        )


class DialogueTopDataLoader:
    def __init__(
        self, data_path="data/dialoglue/top/train.txt", intent_label_to_idx=None
    ):
        self.data_path = data_path
        print(f"Loading data from {self.data_path}")
        dataset = DialoglueTOPDataset(self.data_path, intent_label_to_idx)

    def get_dataloader(
        self, batch_size=4, shuffle=True, tokenizer: AutoTokenizer = None
    ):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tokenizer),
        )
