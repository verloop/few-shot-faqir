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

    datasets = [
        {"source": "haptik", "data": "curekart", "data_subset": "train", "labels": 28},
        {
            "source": "haptik",
            "data": "powerplay11",
            "data_subset": "train",
            "labels": 59,
        },
        {
            "source": "haptik",
            "data": "sofmattress",
            "data_subset": "train",
            "labels": 21,
        },
        {
            "source": "haptik",
            "data": "curekart",
            "data_subset": "subset_train",
            "labels": 28,
        },
        {
            "source": "haptik",
            "data": "powerplay11",
            "data_subset": "subset_train",
            "labels": 59,
        },
        {
            "source": "haptik",
            "data": "sofmattress",
            "data_subset": "subset_train",
            "labels": 21,
        },
        {
            "source": "dialoglue",
            "data": "banking",
            "data_subset": "train",
            "labels": 77,
        },
        {"source": "dialoglue", "data": "clinc", "data_subset": "train", "labels": 150},
        {"source": "dialoglue", "data": "hwu", "data_subset": "train", "labels": 64},
        {
            "source": "dialoglue",
            "data": "banking",
            "data_subset": "train_5",
            "labels": 77,
        },
        {
            "source": "dialoglue",
            "data": "clinc",
            "data_subset": "train_5",
            "labels": 150,
        },
        {"source": "dialoglue", "data": "hwu", "data_subset": "train_5", "labels": 64},
        {
            "source": "dialoglue",
            "data": "banking",
            "data_subset": "train_10",
            "labels": 77,
        },
        {
            "source": "dialoglue",
            "data": "clinc",
            "data_subset": "train_10",
            "labels": 150,
        },
        {"source": "dialoglue", "data": "hwu", "data_subset": "train_10", "labels": 64},
    ]

    for dataset in datasets:
        config["DATASETS"]["DATA_SUBSET"] = dataset["data_subset"]
        config["DATASETS"]["N_LABELS"] = dataset["labels"]
        if dataset["source"] == "haptik" or dataset["data"] == "clinc":
            config["EVALUATION"]["CHECK_OOS_ACCURACY"] = True
            config["DATASETS"]["N_LABELS"] = dataset["labels"] + 1
            if dataset["source"] == "haptik":
                config["DATASETS"]["OOS_CLASS_NAME"] = "NO_NODES_DETECTED"
            else:
                config["DATASETS"]["OOS_CLASS_NAME"] = "oos"
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

        # Evaluate Dense embedding - bert-base-uncased as a feature extractor
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
        config["EVALUATION"]["MODEL_NAME"] = "bert-base-uncased"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="bert-base",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Dense embedding - ConvBERT as a feature extractor
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
        config["EVALUATION"]["MODEL_NAME"] = "models/convbert"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="convbert",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Dense embedding - all-MiniLM-L12-v2 as a feature extractor
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
        config["EVALUATION"]["MODEL_NAME"] = "sentence-transformers/all-MiniLM-L12-v2"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="all-MiniLM-L12-v2",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Dense embedding - all-mpnet-base-v2 as a feature extractor
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
        config["EVALUATION"]["MODEL_NAME"] = "sentence-transformers/all-mpnet-base-v2"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="all-mpnet-base-v2",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Dense embedding - all-MiniLM-L6-v2 as a feature extractor
        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
        config["EVALUATION"]["MODEL_NAME"] = "sentence-transformers/all-MiniLM-L6-v2"
        eval_metrics = evaluate(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="all-MiniLM-L6-v2",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate after finetuning dense embedding - all-mpnet-base-v2
        config["TRAINING"]["MODEL_TYPE"] = "BI_ENCODER"
        config["TRAINING"]["MODEL_NAME"] = "sentence-transformers/all-mpnet-base-v2"
        config["TRAINING"]["TOKENIZER_NAME"] = config["TRAINING"]["MODEL_NAME"]
        config["TRAINING"]["LAYERS_TO_UNFREEZE"] = [11]
        config["TRAINING"]["NUM_ITERATIONS"] = 10
        config["TRAINING"]["SCHEDULER"] = "WarmupLinear"
        config["TRAINING"]["VALIDATION_SPLIT"] = 0.2

        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EVALUATION"]["TOKENIZER_NAME"] = config["TRAINING"]["TOKENIZER_NAME"]
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"

        eval_metrics = train(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="all-mpnet-base-v2_finetuned_10K",
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
        config["TRAINING"]["NUM_ITERATIONS"] = 10
        config["TRAINING"]["SCHEDULER"] = "WarmupLinear"
        config["TRAINING"]["VALIDATION_SPLIT"] = 0.2

        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
        config["EVALUATION"]["TOKENIZER_NAME"] = config["TRAINING"]["TOKENIZER_NAME"]

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
        config["TRAINING"]["SCHEDULER"] = "WarmupLinear"
        config["TRAINING"]["LAYERS_TO_UNFREEZE"] = [5]
        config["TRAINING"]["NUM_ITERATIONS"] = 20000
        config["TRAINING"]["VALIDATION_SPLIT"] = 0.2

        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] = False
        config["EMBEDDINGS"]["EMBEDDING_TYPE"] = "dense"
        config["EVALUATION"]["TOKENIZER_NAME"] = config["TRAINING"]["TOKENIZER_NAME"]

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
        config["EVALUATION"]["TOKENIZER_NAME"] = config["TRAINING"]["TOKENIZER_NAME"]

        eval_metrics = train(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="cross_encoder_bert_finetuned_10K",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate Sentence Bert Cross Encoder model
        config["TRAINING"]["MODEL_TYPE"] = "SBERT_CROSS_ENCODER"
        config["TRAINING"]["MODEL_NAME"] = "cross-encoder/stsb-distilroberta-base"
        config["TRAINING"]["TOKENIZER_NAME"] = config["TRAINING"]["MODEL_NAME"]
        config["TRAINING"]["LAYERS_TO_UNFREEZE"] = [5]
        config["TRAINING"]["SCHEDULER"] = "WarmupLinear"
        config["TRAINING"]["VALIDATION_SPLIT"] = 0.2
        config["TRAINING"]["NUM_ITERATIONS"] = 10

        config["EVALUATION"]["EVALUATION_METHOD"] = "SBERT_CROSS_ENCODER"
        config["EVALUATION"]["TOKENIZER_NAME"] = config["TRAINING"]["TOKENIZER_NAME"]

        eval_metrics = train(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="cross_encoder_sent_bert_finetuned_10K",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate simple finetuned classifier - bert
        config["TRAINING"]["MODEL_NAME"] = "bert-base-uncased"
        config["TRAINING"]["TOKENIZER_NAME"] = config["TRAINING"]["MODEL_NAME"]
        config["TRAINING"]["NUM_ITERATIONS"] = 10
        config["TRAINING"]["VALIDATION_SPLIT"] = 0.2
        config["TRAINING"]["MODEL_TYPE"] = "CLASSIFIER"

        config["EVALUATION"]["EVALUATION_METHOD"] = "CLASSIFIER"
        config["EVALUATION"]["TOKENIZER_NAME"] = config["TRAINING"]["TOKENIZER_NAME"]

        eval_metrics = train(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="bert-classifier_finetuned_10K",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        # Evaluate simple finetuned classifier as embeddings - ConvBERT
        config["TRAINING"]["MODEL_NAME"] = "models/convbert"
        config["TRAINING"]["NUM_ITERATIONS"] = 10
        config["TRAINING"]["VALIDATION_SPLIT"] = 0.2
        config["TRAINING"]["TOKENIZER_NAME"] = config["TRAINING"]["MODEL_NAME"]
        config["TRAINING"]["NUM_ITERATIONS"] = 10
        config["TRAINING"]["VALIDATION_SPLIT"] = 0.2
        config["TRAINING"]["MODEL_TYPE"] = "CLASSIFIER"

        config["EVALUATION"]["EVALUATION_METHOD"] = "EMBEDDINGS"
        config["EVALUATION"]["TOKENIZER_NAME"] = config["TRAINING"]["TOKENIZER_NAME"]

        eval_metrics = train(config)
        eval_metrics_pd = parse_eval_metrics(
            eval_metrics,
            method="convbert-classifier_embedding_finetuned_10K",
            data_source=dataset["source"],
            data_name=dataset["data"],
            config=config,
        )
        evaluation_metrics = pd.concat((evaluation_metrics, eval_metrics_pd))

        evaluation_metrics.to_csv("results.csv", index=False)

    evaluation_metrics.to_csv("results.csv", index=False)


if __name__ == "__main__":
    evaluate_all()
