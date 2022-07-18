import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score

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


def get_dataloader_class(config):
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
    batch_output = []
    for x in dl_data:
        batch_output.extend(x[2])
    return batch_output


def run_evaluation_metrics(
    config, actual_labels, predicted_labels, pred_scores, oos_label_indx
):
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
            pred_labels = predicted_labels[:]
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
        for k in k_vals:
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
                    # change predicted label to oos based on threshold
                    for i, score in enumerate(oos_scores):
                        if score < threshold:
                            oos_predicted[i] = [oos_label_indx] * len(oos_predicted[i])
                    oos_accuracy = success_rate_at_k_batch(oos_actual, oos_predicted, k)
                eval_metrics[k].update({"oos_accuracy": oos_accuracy})
        eval_metrics_thresholds[threshold] = eval_metrics
    return eval_metrics_thresholds


def save_yaml(config, save_dir):
    with open(save_dir + "config.yml", "w") as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)
