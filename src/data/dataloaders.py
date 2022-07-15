from urllib.parse import quote_plus

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from src.data.data_reader import (  # isort:skip
    DialoglueIntentDataset,
    DialoglueTOPDataset,
    HapticDataset,
    QuestionPairDataset,
    QuestionPairSentBertDataset,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collate_batch(batch, tokenizer: AutoTokenizer = None, is_qp=False):
    label_list, text_list = [], []

    if not is_qp:
        for example in batch:
            label_list.append(example["Label"])
            text_list.append(example["Text"])

        if tokenizer:
            batch_output = tokenizer(
                text_list, padding=True, truncation=True, return_tensors="pt"
            )
            label_list = torch.tensor(label_list, dtype=torch.int64)
        else:
            batch_output = text_list
            label_list = np.array(label_list)

        if tokenizer:
            batch_output.to(device)
            label_list.to(device)
    else:
        for example in batch:
            label_list.append(example.label)
            text_list.append(example.texts)

        if tokenizer:
            batch_output = tokenizer.batch_encode_plus(
                text_list, padding=True, truncation=True, return_tensors="pt"
            )
            label_list = torch.tensor(label_list, dtype=torch.int64)
        else:
            batch_output = text_list
            label_list = np.array(label_list)

        if tokenizer:
            batch_output.to(device)
            label_list.to(device)

    return batch_output, label_list, text_list


class HaptikDataLoader:
    def __init__(
        self,
        data_source="haptik",
        dataset_name="curekart",
        data_type="train",
        intent_label_to_idx=None,
    ):
        self.data_path = (
            f"data/{data_source}/{data_type}/{dataset_name}_{data_type}.csv"
        )
        self.qp_data_path = f"data/{data_source}/{data_type}/{dataset_name}_{data_type}_question_pairs.csv"
        print(f"Loading data from {self.data_path}")
        self.dataset = HapticDataset(self.data_path, intent_label_to_idx)
        self.qp_dataset = QuestionPairSentBertDataset(self.qp_data_path)
        train_split = int(0.8 * len(self.qp_dataset))
        test_split = len(self.qp_dataset) - train_split
        self.qp_train_dataset, self.qp_test_dataset = random_split(
            self.qp_dataset,
            [train_split, test_split],
            generator=torch.Generator().manual_seed(2022),
        )

    def get_dataloader(
        self, batch_size=4, shuffle=True, tokenizer: AutoTokenizer = None
    ):
        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tokenizer),
        )

    def get_qp_dataloader(
        self, batch_size=16, shuffle=True, tokenizer: AutoTokenizer = None, is_qp=True
    ):

        return DataLoader(
            self.qp_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
        )

    def get_qp_train_dataloader(
        self, batch_size=16, shuffle=True, tokenizer: AutoTokenizer = None, is_qp=True
    ):

        return DataLoader(
            self.qp_train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
        )

    def get_qp_val_dataloader(
        self, batch_size=16, shuffle=True, tokenizer: AutoTokenizer = None, is_qp=True
    ):

        return DataLoader(
            self.qp_test_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
        )

    def get_qp_sbert_dataloader(self, batch_size=4, shuffle=True):
        self.qp_dataset = QuestionPairSentBertDataset(self.qp_data_path)
        return DataLoader(
            self.qp_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )


class DialogueIntentDataLoader:
    def __init__(
        self,
        data_source="dialoglue",
        dataset_name="banking",
        data_type="train",
        intent_label_to_idx=None,
    ):
        self.data_path = f"data/{data_source}/{dataset_name}/{data_type}.csv"
        self.qp_data_path = (
            f"data/{data_source}/{dataset_name}/{data_type}_question_pairs.csv"
        )
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

    def get_qp_dataloader(self, batch_size=4, shuffle=True):
        self.qp_dataset = QuestionPairDataset(self.qp_data_path)

        return DataLoader(
            self.qp_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def get_qp_sbert_dataloader(self, batch_size=4, shuffle=True):
        self.qp_dataset = QuestionPairSentBertDataset(self.qp_data_path)

        return DataLoader(
            self.qp_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )


class DialogueTopDataLoader:
    def __init__(
        self,
        data_source="dialoglue",
        dataset_name="top",
        data_type="train",
        intent_label_to_idx=None,
    ):
        self.data_path = f"data/{data_source}/{dataset_name}/{data_type}.txt"
        self.qp_data_path = (
            f"data/{data_source}/{dataset_name}/{data_type}_question_pairs.csv"
        )

        print(f"Loading data from {self.data_path}")
        self.dataset = DialoglueTOPDataset(self.data_path, intent_label_to_idx)

    def get_dataloader(
        self, batch_size=4, shuffle=True, tokenizer: AutoTokenizer = None
    ):

        return DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tokenizer),
        )

    def get_qp_dataloader(self, batch_size=4, shuffle=True):
        self.qp_dataset = QuestionPairDataset(self.qp_data_path)

        return DataLoader(
            self.qp_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
