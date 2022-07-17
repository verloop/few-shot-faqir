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
    QuestionPairTestTrainDataset,
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
            label_list.append(list(example[2:]))
            text_list.append([example[0], example[1]])

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

    def train_test_split(self, dataset, val_split_pct):
        train_split = int((1 - val_split_pct) * len(dataset))
        val_split = len(dataset) - train_split
        train_dataset, val_dataset = random_split(
            dataset,
            [train_split, val_split],
            generator=torch.Generator().manual_seed(2022),
        )
        return train_dataset, val_dataset

    def get_dataloader(
        self,
        batch_size=4,
        shuffle=True,
        tokenizer: AutoTokenizer = None,
        val_split_pct=0,
    ):
        if val_split_pct == 0:
            train_dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer),
            )
            val_dataloader = None
            return train_dataloader, val_dataloader
        else:
            train_dataset, val_dataset = self.train_test_split(
                self.dataset, val_split_pct
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer),
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer),
            )
            return train_dataloader, val_dataloader

    def get_qp_dataloader(
        self,
        batch_size=16,
        shuffle=True,
        tokenizer: AutoTokenizer = None,
        is_qp=True,
        val_split_pct=0,
    ):
        self.qp_dataset = QuestionPairDataset(self.qp_data_path)
        if val_split_pct == 0:
            train_dataloader = DataLoader(
                self.qp_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
            )
            val_dataloader = None
            return train_dataloader, val_dataloader
        else:
            train_dataset, val_dataset = self.train_test_split(
                self.qp_dataset, val_split_pct
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
            )
            return train_dataloader, val_dataloader

    def get_qp_sbert_dataloader(self, batch_size=4, shuffle=True, val_split_pct=0):
        self.qp_dataset = QuestionPairSentBertDataset(self.qp_data_path)
        if val_split_pct == 0:
            train_dataloader = DataLoader(
                self.qp_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            val_dataloader = None
            return train_dataloader, val_dataloader
        else:
            train_dataset, val_dataset = self.train_test_split(
                self.qp_dataset, val_split_pct
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            return train_dataloader, val_dataloader

    def get_crossencoder_test_dataloader(
        self, tokenizer: AutoTokenizer = None, is_qp=True, batch_size=4, shuffle=False
    ):
        self.qp_test_train_dataset = QuestionPairTestTrainDataset(self.qp_data_path)
        return DataLoader(
            self.qp_test_train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
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

    def train_test_split(self, dataset, val_split_pct):
        train_split = int((1 - val_split_pct) * len(dataset))
        val_split = len(dataset) - train_split
        train_dataset, val_dataset = random_split(
            dataset,
            [train_split, val_split],
            generator=torch.Generator().manual_seed(2022),
        )
        return train_dataset, val_dataset

    def get_dataloader(
        self,
        batch_size=4,
        shuffle=True,
        tokenizer: AutoTokenizer = None,
        val_split_pct=0,
    ):
        if val_split_pct == 0:
            train_dataloader = DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer),
            )
            val_dataloader = None
            return train_dataloader, val_dataloader
        else:
            train_dataset, val_dataset = self.train_test_split(
                self.dataset, val_split_pct
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer),
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer),
            )
            return train_dataloader, val_dataloader

    def get_qp_dataloader(
        self,
        batch_size=16,
        shuffle=True,
        tokenizer: AutoTokenizer = None,
        is_qp=True,
        val_split_pct=0,
    ):
        self.qp_dataset = QuestionPairDataset(self.qp_data_path)
        if val_split_pct == 0:
            train_dataloader = DataLoader(
                self.qp_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
            )
            val_dataloader = None
            return train_dataloader, val_dataloader
        else:
            train_dataset, val_dataset = self.train_test_split(
                self.qp_dataset, val_split_pct
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
            )
            return train_dataloader, val_dataloader

    def get_qp_sbert_dataloader(self, batch_size=4, shuffle=True, val_split_pct=0):
        self.qp_dataset = QuestionPairSentBertDataset(self.qp_data_path)
        if val_split_pct == 0:
            train_dataloader = DataLoader(
                self.qp_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            val_dataloader = None
            return train_dataloader, val_dataloader
        else:
            train_dataset, val_dataset = self.train_test_split(
                self.qp_dataset, val_split_pct
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
            return train_dataloader, val_dataloader

    def get_crossencoder_test_dataloader(
        self, tokenizer: AutoTokenizer = None, is_qp=True, batch_size=4, shuffle=False
    ):
        self.qp_test_train_dataset = QuestionPairTestTrainDataset(self.qp_data_path)
        return DataLoader(
            self.qp_test_train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=lambda b: collate_batch(b, tokenizer, is_qp),
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
