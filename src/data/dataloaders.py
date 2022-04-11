import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.data.data_reader import (  # isort:skip
    DialoglueIntentDataset,
    DialoglueTOPDataset,
    HapticDataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize_batch(batch, tokenizer="bert"):
    label_list, text_list, = [], []

    if tokenizer == "bert":

        tokenizer = Tokenizer.from_pretrained("bert-base-uncased")

        for example in batch:
            label_list.append(example["Label"])
            processed_text = torch.tensor(
                tokenizer.encode(example["Text"]).ids, dtype=torch.int64
            )
            text_list.append(processed_text)
            text_list = pad_sequence(text_list, batch_first=True)

    else:
        for example in batch:
            label_list.append(example["Label"])
            text_list.append(processed_text)
            text_list = torch.tensor(text_list, dtype=torch.int64)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    return label_list, text_list


def collate_batch(batch):
    # change tokenizer here
    label_list, text_list = tokenize_batch(batch, tokenizer="bert")

    return text_list.to(device), label_list.to(device)


class HaptikDataLoader:
    def __init__(self, data_path="data/haptik/train/curekart_train.csv"):
        self.data_path = data_path
        print(f"Loading data from {self.data_path}")

    def get_dataloader(self, batch_size=4, shuffle=True):
        dataset = HapticDataset(self.data_path)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch
        )


class DialogueIntentDataLoaders:
    def __init__(self, data_path="data/dialoglue/banking/train.csv"):
        self.data_path = data_path
        print(f"Loading data from {self.data_path}")

    def get_dataloader(self, batch_size=4, shuffle=True):
        dataset = DialoglueIntentDataset(self.data_path)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch
        )


class DialogueTopDataLoaders:
    def __init__(self, data_path="data/dialoglue/top/train.txt"):
        self.data_path = data_path
        print(f"Loading data from {self.data_path}")

    def get_dataloader(self, batch_size=4, shuffle=True):
        dataset = DialoglueTOPDataset(self.data_path)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch
        )
