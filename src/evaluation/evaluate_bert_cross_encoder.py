import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.embeddings.sparse_embeddings import SparseEmbedding

from src.utils.utils import (  # isort:skip
    get_dataloader_class,
    run_evaluation_metrics,
)


def evaluate_bert_crossencoder(config):
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
    print("Running evaluation with cross encoders")
    batch_size = config["EVALUATION"]["BATCH_SIZE"]
    tokenizer = AutoTokenizer.from_pretrained(config["EVALUATION"]["TOKENIZER_NAME"])
    # Evaluation on reduced samples
    if config["DATASETS"]["DATASET_SOURCE"] == "haptik":
        data_subset = "subset_train"
    else:
        data_subset = "train_5"
    dl_test = dataloader(
        data_source=data_source,
        dataset_name=dataset_name,
        data_type="test",
        data_subset=data_subset,
        intent_label_to_idx=dl_train.dataset.intent_label_to_idx,
    )
    test_dataloader = dl_test.get_crossencoder_test_dataloader(
        tokenizer=tokenizer, batch_size=batch_size
    )

    base_model_name = config["EVALUATION"][
        "TOKENIZER_NAME"
    ]  # loading the base model using the same as tokenizer
    model_name = config["EVALUATION"]["MODEL_NAME"]
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
    model.to(torch.device(device))

    if os.path.isfile(model_name):
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    with torch.no_grad():
        pred_probs = []
        labels = []
        total_steps = len(test_dataloader)
        steps_done = 0
        progress_bar = tqdm(range(total_steps))
        idx = []
        for batch in test_dataloader:
            batch[0].to(torch.device(device))
            outputs = model(**batch[0])
            class_predictions = F.softmax(outputs.logits, dim=1)
            # for getting probability for positive class 1
            pred_probs += [i.detach().cpu().tolist()[1] for i in class_predictions]
            labels += batch[1][:, 0].detach().cpu().tolist()
            idx += batch[1][:, 1].detach().cpu().tolist()
            steps_done += 1
            progress_bar.update(1)
            # if steps_done == 2:
            #     break

        idx = np.array(idx)
        pred_probs = np.array(pred_probs)
        labels = np.array(labels)
        true_vals = []
        predicted = []
        pred_scores = []
        for i in set(idx):
            # indx is set for a test set record.Get the labels & scores for each test record
            # print("idx",i)
            matched = np.where(idx == i)[0]
            pred_probs_ = pred_probs[matched]
            # print("pred_probs_")
            # print(pred_probs_)
            labels_ = labels[matched]
            # print("labels_")
            # print(labels_)
            pred_prob_argsorted = np.argsort(pred_probs_)[::-1][:10]
            pred_probs_sorted = pred_probs_[pred_prob_argsorted]
            # print("pred_probs_sorted")
            # print(pred_probs_sorted)
            labels_sorted = labels_[pred_prob_argsorted]
            # print(labels_sorted)
            # Count only the first position where the correct answer appears - Can be done if needed
            # Checking oos
            if len(set(labels_)) == 1 and list(set(labels_))[0] == 0:
                # No questions were matching which means OOS - label of 2
                true_vals.append([2] * len(pred_prob_argsorted))
            else:
                true_vals.append([1] * len(pred_prob_argsorted))
            pred_scores.append(
                list(pred_probs_sorted)[0]
            )  # Get only the first score for marking it as OOS
            predicted.append(list(labels_sorted))

    # print(true_vals)
    # print(pred_scores)
    # print(predicted)
    oos_label_indx = 2  # Giving the value of 2 for OOS labels in Similarity method

    eval_metrics = run_evaluation_metrics(
        config, true_vals, predicted, pred_scores, oos_label_indx
    )
    return eval_metrics
