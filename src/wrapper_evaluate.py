import pandas as pd
import yaml

from src.evaluate import evaluate, evaluate_bm25
from src.train import train


def parse_eval_metrics(eval_metrics_thresh, method, data_source, data_name, config):
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
        "method": [method] * len(k_vals),
        "data_source": [data_source] * len(k_vals),
        "dataset": [data_name] * len(k_vals),
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


def evaluate_all():
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    evaluation_metrics = pd.DataFrame({})

    # datasets = [{"source":"haptik","data":"curekart"},{"source":"haptik","data":"powerplay11"},{"source":"haptik","data":"sofmattress"},
    # {"source":"dialoglue","data":"banking"},{"source":"dialoglue","data":"clinc"},{"source":"dialoglue","data":"hwu"}]

    datasets = [
        {"source": "haptik", "data": "curekart", "data_subset": "subset_train"},
        {"source": "dialoglue", "data": "banking", "data_subset": "train_5"}
        # {"source": "haptik", "data": "sofmattress"},
    ]

    for dataset in datasets:
        config["DATASETS"]["DATA_SUBSET"] = dataset["data_subset"]

        if dataset["source"] == "haptik" or dataset["data"] == "clinc":
            config["EVALUATION"]["CHECK_OOS_ACCURACY"] = True
        else:
            config["EVALUATION"]["CHECK_OOS_ACCURACY"] = False
        config["DATASETS"]["DATASET_SOURCE"] = dataset["source"]
        config["DATASETS"]["DATASET_NAME"] = dataset["data"]

        # Evaluate bm25
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = True
        config["EMBEDDINGS"]["SELECT_BM25_FASTTEXT_GLOVE"] = "BM25"
        eval_metrics = evaluate_bm25(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="BM25",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate FastText
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = True
        config["EMBEDDINGS"]["SELECT_BM25_FASTTEXT_GLOVE"] = "FASTTEXT"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="FASTTEXT",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Glove
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = True
        config["EMBEDDINGS"]["SELECT_BM25_FASTTEXT_GLOVE"] = "GLOVE"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="GLOVE",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Sparse embedding - TFIDF - Word
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "sparse"
        config["EMBEDDINGS"]["SPARSE_EMB_METHOD"] = "tfidf-word"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="TFIDF-WORD",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Sparse embedding - TFIDF - Character
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "sparse"
        config["EMBEDDINGS"]["SPARSE_EMB_METHOD"] = "tfidf-char"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="TFIDF-CHAR",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Sparse embedding - CV
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "sparse"
        config["EMBEDDINGS"]["SPARSE_EMB_METHOD"] = "cv"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="CV",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Dense embedding - bert-base-uncased
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
        config["EMBEDDINGS"]["MODEL_NAME"] = "bert-base-uncased"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="bert-base",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Dense embedding - all-MiniLM-L12-v2
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
        config["EMBEDDINGS"]["MODEL_NAME"] = "sentence-transformers/all-MiniLM-L12-v2"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="all-MiniLM-L12-v2",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Dense embedding - all-MiniLM-L6-v2
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
        config["EMBEDDINGS"]["MODEL_NAME"] = "sentence-transformers/all-MiniLM-L6-v2"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="all-MiniLM-L6-v2",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate after finetuning dense embedding - miniLM-L6
        config["TRAINING"]["MODEL_TYPE"] = "BI_ENCODER"
        config["TRAINING"]["MODEL_NAME"] = "sentence-transformers/all-MiniLM-L6-v2"
        config["TRAINING"]["TOKENIZER_NAME"] = config["TRAINING"]["MODEL_NAME"]
        config["TRAINING"]["LAYERS_TO_UNFREEZE"] = [5]
        config["TRAINING"]["NUM_ITERATIONS"] = 10000
        config["TRAINING"]["SCHEDULER"] = "WarmupLinear"
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"

        eval_metrics = train(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="all-MiniLM-L6-v2_finetuned_10K",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate after finetuning dense embedding - miniLM-L6
        config["TRAINING"]["MODEL_TYPE"] = "BI_ENCODER"
        config["TRAINING"]["MODEL_NAME"] = "sentence-transformers/all-MiniLM-L6-v2"
        config["TRAINING"]["TOKENIZER_NAME"] = config["TRAINING"]["MODEL_NAME"]
        config["TRAINING"]["LAYERS_TO_UNFREEZE"] = [5]
        config["TRAINING"]["NUM_ITERATIONS"] = 20000
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"

        eval_metrics = train(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="all-MiniLM-L6-v2_finetuned_20K",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate cross encoder model
        config["TRAINING"]["MODEL_TYPE"] = "CROSS_ENCODER"
        config["TRAINING"]["MODEL_NAME"] = "bert-base-uncased"
        config["TRAINING"]["TOKENIZER_NAME"] = config["TRAINING"]["MODEL_NAME"]
        config["TRAINING"]["LAYERS_TO_UNFREEZE"] = [11]
        config["TRAINING"]["SCHEDULER"] = "linear"
        config["TRAINING"]["VALIDATION_SPLIT"] = 0
        config["TRAINING"]["NUM_ITERATIONS"] = 10
        config["EVALUATION"]["EVALUATION_METHOD"] = "CROSS_ENCODER"

        eval_metrics = train(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="cross_encoder_bert_finetuned_20K",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

    evaluation_metrics.to_csv("results.csv", index=False)


if __name__ == "__main__":
    evaluate_all()
