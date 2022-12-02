import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm.auto import tqdm

from src.utils.utils import (  # isort:skip
    get_dataloader_class,
    run_evaluation_metrics,
)


def evaluate_sbert_crossencoder(config):
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
    print("Running evaluation with sbert cross encoders")
    batch_size = config["EVALUATION"]["BATCH_SIZE"]
    # Evaluation on reduced samples
    if config["DATASETS"]["DATASET_SOURCE"] == "hint3":
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
    test_dataloader = dl_test.get_crossencoder_test_dataloader(batch_size=batch_size)
    model_name = config["EVALUATION"]["MODEL_NAME"]
    model = CrossEncoder(model_name, device=device)

    with torch.no_grad():
        pred_probs = []
        labels = []
        total_steps = len(test_dataloader)
        steps_done = 0
        progress_bar = tqdm(range(total_steps))
        idx = []
        sentence_1 = []
        sentence_2 = []
        for batch in test_dataloader:
            # print(batch[0])
            sentence_1 += [each[0] for each in batch[0]]
            sentence_2 += [each[1] for each in batch[0]]
            sim_predictions = model.predict(batch[0])
            # for getting probability for positive class 1
            pred_probs += list(sim_predictions)
            # labels += list(batch[1][:, 0])
            labels += [each[0] for each in batch[1]]
            # idx += list(batch[1][:, 1])
            idx += [each[1] for each in batch[1]]
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

    test_predictions = pd.DataFrame(
        {
            "sentence_1": sentence_1,
            "sentence_2": sentence_2,
            "idx": idx,
            "actual": labels,
            "pred_score": pred_probs,
        }
    )
    fname = f"{config['EVALUATION']['MODEL_NAME']}_{config['DATASETS']['DATASET_NAME']}"
    fname = fname.replace("/", "")
    test_predictions.to_csv(f"predictions_sbertce_{fname}.csv")

    return eval_metrics
