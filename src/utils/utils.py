import numpy as np
from sklearn.metrics import f1_score

import torch
from src.data.dataloaders import DialogueIntentDataLoader, HaptikDataLoader

from src.utils.metrics import (  # isort:skip
    map_at_k,
    mrr,
    ndcg_at_k_batch,
    precision_at_k_batch,
    success_rate_at_k_batch,
    f1_score_micro_k,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader(config):
    if config["DATASETS"]["DATASET_SOURCE"] == "haptik":
        return HaptikDataLoader
    elif config["DATASETS"]["DATASET_SOURCE"] == "dialoglue":
        return DialogueIntentDataLoader
    else:
        print("Data loader not recognized")
        return


def get_batched_embeddings_dense(dl_data, emb_model):
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


def get_batched_embeddings_sparse(dl_data, emb_model):
    embeddings = []
    labels = []
    texts = []
    for i, x in enumerate(dl_data):
        batch_embeddings = emb_model.get_embeddings(x[0])
        batch_embeddings = batch_embeddings.toarray()
        if len(embeddings) > 0:
            embeddings = np.concatenate((embeddings, batch_embeddings), axis=0)
            labels = np.concatenate((labels, x[1]), axis=0)
        else:
            embeddings = batch_embeddings
            labels = x[1]
        texts = texts + x[2]
    return embeddings, labels, texts


def get_train_text(dl_data):
    batch_output = []
    for x in dl_data:
        batch_output = batch_output + x[0]
    return batch_output


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
