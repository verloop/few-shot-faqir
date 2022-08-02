import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils.utils import (  # isort:skip
    get_dataloader_class,
    run_evaluation_metrics,
)


def evaluate_bert_classifier(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader = get_dataloader_class(config)
    data_source = config["DATASETS"]["DATASET_SOURCE"]
    dataset_name = config["DATASETS"]["DATASET_NAME"]
    data_subset = config["DATASETS"]["DATA_SUBSET"]

    dl_train = dataloader(
        data_source=data_source,
        dataset_name=dataset_name,
        data_type="train",
        data_subset=data_subset,
    )
    dl_test = dataloader(
        data_source=data_source,
        dataset_name=dataset_name,
        data_type="test",
        data_subset="test",
        intent_label_to_idx=dl_train.dataset.intent_label_to_idx,
    )
    print("Running evaluation with classifier approach")
    num_labels = config["DATASETS"]["N_LABELS"]
    model_name = config["EVALUATION"]["MODEL_NAME"]
    batch_size = config["EVALUATION"]["BATCH_SIZE"]
    tokenizer = AutoTokenizer.from_pretrained(config["EVALUATION"]["TOKENIZER_NAME"])
    test_dataloader, _ = dl_test.get_dataloader(
        batch_size=batch_size, tokenizer=tokenizer
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    model.to(device)
    model.eval()
    prediction_labels_test = []
    true_labels_test = []
    pred_probs = []
    for batch in test_dataloader:
        true_labels_test += batch[1].numpy().flatten().tolist()
        with torch.no_grad():
            batch[0].to(torch.device(device))
            outputs = model(**batch[0])
            prediction_probs = F.softmax(outputs.logits, dim=1)
            predicted = torch.argsort(outputs.logits, 1, descending=True)
            predicted = predicted.detach().cpu().numpy()
            prediction_probs = prediction_probs.detach().cpu().numpy()
            pred_probs_ = [list(i[j])[0] for i, j in zip(prediction_probs, predicted)]
            pred_probs += pred_probs_
            prediction_labels_test += list(predicted)

    true_labels_test = [[i] for i in true_labels_test]

    oos_label_indx = None
    if config["EVALUATION"]["CHECK_OOS_ACCURACY"]:
        oos_label_indx = dl_train.dataset.intent_label_to_idx[
            config["DATASETS"]["OOS_CLASS_NAME"]
        ]

    eval_metrics = run_evaluation_metrics(
        config, true_labels_test, prediction_labels_test, pred_probs, oos_label_indx
    )
    return eval_metrics
