import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

import torch
import yaml
from src.data.dataloaders import DialogueIntentDataLoader, HaptikDataLoader
from src.embeddings.dense_embeddings import DenseEmbeddings, get_similar
from transformers import AutoTokenizer

from src.utils.metrics import (  # isort:skip
    map_at_k,
    mrr,
    ndcg_at_k_batch,
    precision_at_k_batch,
    success_rate_at_k_batch,
    f1_score_micro_k,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("src/config/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


def get_dataloader(config):
    if config["DATASETS"]["DATASET_SOURCE"] == "haptik":
        return HaptikDataLoader
    elif config["DATASETS"]["DATASET_SOURCE"] == "dialoglue":
        return DialogueIntentDataLoader
    else:
        print("Data loader not recognized")
        return


def get_batched_embeddings(dl_data, emb_model):
    embeddings = torch.empty(0)
    labels = []
    texts = []
    for i, x in enumerate(dl_data):
        batch_embeddings = emb_model.get_embeddings(x[0].to(device))
        if embeddings.size()[0] > 0:
            embeddings = torch.cat([embeddings, batch_embeddings], dim=0)
            labels = torch.cat([labels, x[1]], dim=0)
            texts = texts + x[2]
        else:
            embeddings = batch_embeddings
            labels = x[1]
            texts = x[2]
    return embeddings.cpu().detach().numpy(), labels.cpu().detach().numpy(), texts


def run_evaluation_metrics(config, test_labels_, pred_labels):
    k_vals = config["EVALUATION"]["K_VAL"]
    for k in k_vals:
        pred_labels_k = [x[:k] for x in pred_labels]
        if config["EVALUATION"]["CHECK_SUCCESS_RATE"]:
            sr = success_rate_at_k_batch(test_labels_, pred_labels_k, k)
            print(f"SR @ {k} is {sr}")
        if config["EVALUATION"]["CHECK_PRECISION"]:
            precision = precision_at_k_batch(test_labels_, pred_labels_k, k)
            print(f"Precision @ {k} is {precision}")
        if config["EVALUATION"]["CHECK_MAP"]:
            map_k = map_at_k(test_labels_, pred_labels_k, k)
            print(f"MAP @ {k} is {map_k}")
        if config["EVALUATION"]["CHECK_NDCG"]:
            ndcg_k = ndcg_at_k_batch(test_labels_, pred_labels_k, k)
            print(f"NDCG @ {k} is {ndcg_k}")
        if config["EVALUATION"]["CHECK_MRR"]:
            mrr_val = mrr(test_labels_, pred_labels_k)
            print(f"MRR is {mrr_val}")
        if config["EVALUATION"]["CHECK_F1_MICRO"]:
            f1_val = f1_score_micro_k(test_labels_, pred_labels_k, k)
            print(f"F1 micro is {f1_val}")
        if k == 1:
            actual = [each[0] for each in test_labels_]
            predicted = [each[0] for each in pred_labels_k]
            if config["EVALUATION"]["CHECK_F1_MACRO"]:
                f1_val = f1_score(actual, predicted, average="macro")
                print(f"F1 macro is {f1_val}")
            if config["EVALUATION"]["CHECK_F1_WEIGHTED"]:
                f1_val = f1_score(actual, predicted, average="weighted")
                print(f"F1 weighted is {f1_val}")


dataloader = get_dataloader(config)
data_source = config["DATASETS"]["DATASET_SOURCE"]
dataset_name = config["DATASETS"]["DATASET_NAME"]
dl_train = dataloader(
    data_source=data_source, dataset_name=dataset_name, data_type="train"
)
dl_test = dataloader(
    data_source=data_source,
    dataset_name=dataset_name,
    data_type="test",
    intent_label_to_idx=dl_train.dataset.intent_label_to_idx,
)

model_name = config["EMBEDDINGS"]["MODEL_NAME"]

model_tokenizer = AutoTokenizer.from_pretrained(model_name)
dl_train_data = dl_train.get_dataloader(tokenizer=model_tokenizer)
dl_test_data = dl_test.get_dataloader(tokenizer=model_tokenizer)

de = DenseEmbeddings(model_name)

train_embeddings, train_labels, train_texts = get_batched_embeddings(dl_train_data, de)
test_embeddings, test_labels, test_texts = get_batched_embeddings(dl_test_data, de)

test_labels_ = [[each] for each in test_labels]

pred_labels = []
pred_scores = []

for each in test_embeddings:
    indx, scores = get_similar(train_embeddings, each, top_k=10)
    oos_flag = True if scores[0] < config["EMBEDDINGS"]["OOS_THRESHOLD"] else False
    predicted_labels = train_labels[indx]
    if oos_flag and config["DATASETS"]["DATASET_SOURCE"] == "haptik":
        oos_class_name = config["DATASETS"]["OOS_CLASS_NAME"]
        predicted_labels[0] = dl_train.dataset.intent_label_to_idx[oos_class_name]
    pred_labels.append(predicted_labels)
    pred_scores.append(scores[0])

run_evaluation_metrics(config, test_labels_, pred_labels)

# For debugging and checking results. Remove later

pred_label_names = [
    dl_train.dataset.intent_idx_to_label[x[0]] for x in list(pred_labels)
]
test_label_names = [dl_train.dataset.intent_idx_to_label[x] for x in list(test_labels)]
test_predictions = pd.DataFrame(
    {
        "text": test_texts,
        "actual": test_label_names,
        "predicted": pred_label_names,
        "pred_score": pred_scores,
    }
)
test_predictions.to_csv("test_predictions.csv")
