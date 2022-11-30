import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import f1_score

from src.data.dataloaders import DialogueIntentDataLoader, HintDataLoader

from src.utils.metrics import (  # isort:skip
    map_at_k,
    mrr,
    ndcg_at_k_batch,
    precision_at_k_batch,
    success_rate_at_k_batch,
    f1_score_micro_k,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloader_class(config):
    """
    Helper function to return the dataset specific dataloader
    """
    if config["DATASETS"]["DATASET_SOURCE"] == "hint3":
        return HintDataLoader
    elif config["DATASETS"]["DATASET_SOURCE"] == "dialoglue":
        return DialogueIntentDataLoader
    else:
        print("Data loader not recognized")
        return


def get_batched_embeddings_dense(dl_data, emb_model):
    """
    Helper function to return dense embeddings using the embedding model passed. Should be an object of the DenseEmbeddings class
    """
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
    """
    Helper function to return dense embeddings using the embedding model passed. Should be an object of the SparseEmbedding class
    """
    embeddings = []
    labels = []
    texts = []
    for i, x in enumerate(dl_data):
        batch_embeddings = emb_model.get_embeddings(x[0])
        if type(batch_embeddings) != np.ndarray:
            batch_embeddings = batch_embeddings.toarray()
        if len(embeddings) > 0:
            embeddings = np.concatenate((embeddings, batch_embeddings), axis=0)
            labels = np.concatenate((labels, x[1]), axis=0)
        else:
            embeddings = batch_embeddings
            labels = x[1]
        texts = texts + x[2]
    return embeddings, labels, texts


def get_text_from_dl(dl_data):
    """
    Helper function to get text from the data loader
    """
    batch_output = []
    for x in dl_data:
        batch_output.extend(x[2])
    return batch_output


def run_evaluation_metrics(
    config, actual_labels, predicted_labels, pred_scores, oos_label_indx
):
    """
    Wrapper method to run evaluation metrics
    Arguments
    ----------
    config: config from config.yaml
    actual_labels: ground truth encoded labels
    predicted_labels : predicted labels from the model
    pred_scores : predicted scores for generating scores with different OOS thresholds
    oos_label_indx: label index of the class which is marked Out of Scope

    Returns
    --------
    A dictionary with evaluation metrics based on settings in config

    """
    k_vals = config["EVALUATION"]["K_VAL"]
    eval_metrics_thresholds = {}
    for threshold in config["EVALUATION"]["OOS_THRESHOLD"]:
        eval_metrics = {}
        if oos_label_indx:
            # remove out of scope intents based on ground truth to get inscope accuracy
            indx_in_scope = [
                i for i, label in enumerate(actual_labels) if label[0] != oos_label_indx
            ]
            indx_out_scope = [
                i for i, label in enumerate(actual_labels) if label[0] == oos_label_indx
            ]
        for k in k_vals:
            pred_labels = predicted_labels[:]
            if oos_label_indx and k == 1:
                # change predicted label to oos based on threshold
                for i, score in enumerate(pred_scores):
                    if score < threshold:
                        pred_labels[i] = [oos_label_indx] * len(pred_labels[i])
            if oos_label_indx:
                # separate the sets for in scope and out of scope accuracy
                oos_predicted = [pred_labels[i] for i in indx_out_scope]
                oos_actual = [actual_labels[i] for i in indx_out_scope]
                oos_scores = [pred_scores[i] for i in indx_out_scope]
                gt_labels = [actual_labels[i] for i in indx_in_scope]
                pred_labels = [pred_labels[i] for i in indx_in_scope]
            else:
                gt_labels = actual_labels
                pred_labels = predicted_labels
                oos_actual, oos_predicted, oos_scores = None, None, None
            pred_labels_k = [x[:k] for x in pred_labels]
            eval_metrics[k] = {}
            if config["EVALUATION"]["CHECK_SUCCESS_RATE"]:
                sr = success_rate_at_k_batch(gt_labels, pred_labels_k, k)
                eval_metrics[k].update({"sr": sr})
                print(f"SR @ {k} is {sr}")
            if config["EVALUATION"]["CHECK_PRECISION"]:
                precision = precision_at_k_batch(gt_labels, pred_labels_k, k)
                eval_metrics[k].update({"precision": precision})
                print(f"Precision @ {k} is {precision}")
            if config["EVALUATION"]["CHECK_MAP"]:
                map_k = map_at_k(gt_labels, pred_labels_k, k)
                eval_metrics[k].update({"map": map_k})
                print(f"MAP @ {k} is {map_k}")
            if config["EVALUATION"]["CHECK_NDCG"]:
                ndcg_k = ndcg_at_k_batch(gt_labels, pred_labels_k, k)
                eval_metrics[k].update({"ndcg": ndcg_k})
                print(f"NDCG @ {k} is {ndcg_k}")
            if config["EVALUATION"]["CHECK_MRR"]:
                mrr_val = mrr(gt_labels, pred_labels_k)
                eval_metrics[k].update({"mrr": mrr_val})
                print(f"MRR is {mrr_val}")
            if config["EVALUATION"]["CHECK_F1_MICRO"]:
                f1_micro = f1_score_micro_k(gt_labels, pred_labels_k, k)
                eval_metrics[k].update({"f1_micro": f1_micro})
                print(f"F1 micro is {f1_micro}")
            if k == 1:
                actual = [each[0] for each in gt_labels]
                predicted = [each[0] for each in pred_labels_k]
            if config["EVALUATION"]["CHECK_F1_MACRO"]:
                f1_macro = -1
                if k == 1:
                    f1_macro = f1_score(actual, predicted, average="macro")
                    print(f"F1 macro is {f1_macro}")
                eval_metrics[k].update({"f1_macro": f1_macro})
            if config["EVALUATION"]["CHECK_F1_WEIGHTED"]:
                f1_weighted = -1
                if k == 1:
                    f1_weighted = f1_score(actual, predicted, average="weighted")
                    print(f"F1 weighted is {f1_weighted}")
                eval_metrics[k].update({"f1_weighted": f1_weighted})
            if config["EVALUATION"]["CHECK_OOS_ACCURACY"]:
                oos_accuracy = -1
                if oos_actual and oos_predicted and k == 1:
                    oos_accuracy = success_rate_at_k_batch(oos_actual, oos_predicted, k)
                eval_metrics[k].update({"oos_accuracy": oos_accuracy})
        eval_metrics_thresholds[threshold] = eval_metrics
    return eval_metrics_thresholds


def save_yaml(config, save_dir):
    with open(save_dir + "config.yml", "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)


def parse_eval_metrics(eval_metrics_thresh, config):
    """
    Helper function to save evaluation metrics as a csv file
    """
    (
        thresholds,
        k_vals,
        sr,
        mrr,
        ndcg,
        precision,
        map_val,
        f1_micro,
        f1_macro,
        f1_weighted,
        oos_accuracy,
    ) = ([], [], [], [], [], [], [], [], [], [], [])
    for thresh in eval_metrics_thresh:
        eval_metrics = eval_metrics_thresh[thresh]
        for k in eval_metrics:
            thresholds.append(thresh)
            k_vals.append(k)
            if "sr" in eval_metrics[k]:
                sr.append(eval_metrics[k]["sr"])
            else:
                sr.append(-1)
            if "mrr" in eval_metrics[k]:
                mrr.append(eval_metrics[k]["mrr"])
            else:
                mrr.append(-1)
            if "ndcg" in eval_metrics[k]:
                ndcg.append(eval_metrics[k]["ndcg"])
            else:
                ndcg.append(-1)
            if "precision" in eval_metrics[k]:
                precision.append(eval_metrics[k]["precision"])
            else:
                precision.append(-1)
            if "map" in eval_metrics[k]:
                map_val.append(eval_metrics[k]["map"])
            else:
                map_val.append(-1)
            if "f1_micro" in eval_metrics[k]:
                f1_micro.append(eval_metrics[k]["f1_micro"])
            else:
                f1_micro.append(-1)
            if "f1_macro" in eval_metrics[k]:
                f1_macro.append(eval_metrics[k]["f1_macro"])
            else:
                f1_macro.append(-1)
            if "f1_weighted" in eval_metrics[k]:
                f1_weighted.append(eval_metrics[k]["f1_weighted"])
            else:
                f1_weighted.append(-1)
            if "oos_accuracy" in eval_metrics[k]:
                oos_accuracy.append(eval_metrics[k]["oos_accuracy"])
            else:
                oos_accuracy.append(-1)
    results = {
        "data_source": [config["DATASETS"]["DATASET_SOURCE"]] * len(k_vals),
        "dataset": [config["DATASETS"]["DATASET_NAME"]] * len(k_vals),
        "subset": [config["DATASETS"]["DATA_SUBSET"]] * len(k_vals),
        "threshold": thresholds,
        "k": k_vals,
        "sr": sr,
        "precision": precision,
        "mrr": mrr,
        "ndcg": ndcg,
        "map": map_val,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "oos_accuracy": oos_accuracy,
    }
    data = pd.DataFrame(results)
    return data
