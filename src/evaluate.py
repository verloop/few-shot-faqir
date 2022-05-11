import re

import numpy as np
import pandas as pd
import rank_bm25
import spacy
import yaml
from transformers import AutoTokenizer

from src.embeddings.dense_embeddings import DenseEmbeddings, get_similar
from src.embeddings.sparse_embeddings import SparseEmbedding

from src.utils.utils import (  # isort:skip
    get_batched_embeddings_dense,
    get_batched_embeddings_sparse,
    get_dataloader_class,
    get_text_from_dl,
    run_evaluation_metrics,
)


def evaluate(config):
    dataloader = get_dataloader_class(config)
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

    if config["EMBEDDINGS"]["EMBEDDING_TYPE"] == "dense":
        model_name = config["EMBEDDINGS"]["MODEL_NAME"]
        model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        dl_train_data = dl_train.get_dataloader(tokenizer=model_tokenizer)
        dl_test_data = dl_test.get_dataloader(shuffle=False, tokenizer=model_tokenizer)
        emb_model = DenseEmbeddings(model_name)
        train_embeddings, train_labels, train_texts = get_batched_embeddings_dense(
            dl_train_data, emb_model
        )
        test_embeddings, test_labels, test_texts = get_batched_embeddings_dense(
            dl_test_data, emb_model
        )
    else:
        dl_train_data = dl_train.get_dataloader()
        dl_test_data = dl_test.get_dataloader(shuffle=False)
        emb_model = SparseEmbedding(
            sparse_embedding_method=config["EMBEDDINGS"]["SPARSE_EMB_METHOD"]
        )
        train_data = get_text_from_dl(dl_train_data)
        emb_model.train(train_data)
        train_embeddings, train_labels, train_texts = get_batched_embeddings_sparse(
            dl_train_data, emb_model
        )
        test_embeddings, test_labels, test_texts = get_batched_embeddings_sparse(
            dl_test_data, emb_model
        )

    test_labels_ = [[label] for label in test_labels]

    pred_labels = []
    pred_scores = []

    for test_embedding in test_embeddings:
        indx, scores = get_similar(train_embeddings, test_embedding, top_k=10)
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
    test_label_names = [
        dl_train.dataset.intent_idx_to_label[x] for x in list(test_labels)
    ]
    test_predictions = pd.DataFrame(
        {
            "text": test_texts,
            "actual": test_label_names,
            "predicted": pred_label_names,
            "pred_score": pred_scores,
        }
    )
    test_predictions.to_csv("test_predictions.csv")


def evaluate_bm25(config):
    nlp = spacy.load("en_core_web_sm")
    tokenizer = nlp.tokenizer

    def _preprocess_text(sentence):
        # remove punctuation
        sentence = re.sub(r"[^\w\s]", "", sentence)
        # tokenize
        tokens = tokenizer(sentence)
        # lemmatize and remove stop words
        preprocessed_text = []
        for tok in tokens:
            if not tok.is_stop:
                if tok.lemma_:
                    preprocessed_text.append(tok.lemma_)
                else:
                    preprocessed_text.append(tok)
        return preprocessed_text

    dataloader = get_dataloader_class(config)
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
    dl_train_data = dl_train.get_dataloader(batch_size=100, shuffle=False)
    dl_test_data = dl_test.get_dataloader(batch_size=1, shuffle=False)
    train_data = get_text_from_dl(dl_train_data)

    # Get train and test labels
    train_labels = []
    for batch in dl_train_data:
        for label in batch[1]:
            train_labels.append(label)

    train_labels = np.array(train_labels)

    preprocessed_train_data = [_preprocess_text(text) for text in train_data]
    # print(preprocessed_train_data)
    bm25 = rank_bm25.BM25Okapi(preprocessed_train_data)

    top_k = 10
    preds, test_labels, test_labels_, pred_labels, pred_scores, test_texts = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    for _, label_batch, text_batch in dl_test_data:
        test_texts.append(text_batch[0])
        text = _preprocess_text(text_batch[0])
        sims = bm25.get_scores(text)
        most_similar_idxs = np.argsort(sims)[::-1][:top_k]
        most_similar_scores = np.sort(sims)[::-1][:top_k]

        oos_flag = (
            True
            if most_similar_scores[0] < config["EMBEDDINGS"]["OOS_THRESHOLD"]
            else False
        )
        predicted_labels = train_labels[list(most_similar_idxs)]
        if oos_flag and config["DATASETS"]["DATASET_SOURCE"] == "haptik":
            oos_class_name = config["DATASETS"]["OOS_CLASS_NAME"]
            predicted_labels[0] = dl_train.dataset.intent_label_to_idx[oos_class_name]
        pred_labels.append(predicted_labels)
        pred_scores.append(most_similar_scores[0])
        test_labels_.append(list(label_batch))
        test_labels.append(label_batch[0])
        preds.append(most_similar_scores)

    run_evaluation_metrics(config, test_labels_, pred_labels)

    # For debugging and checking results. Remove later

    pred_label_names = [
        dl_train.dataset.intent_idx_to_label[x[0]] for x in list(pred_labels)
    ]
    test_label_names = [
        dl_train.dataset.intent_idx_to_label[x] for x in list(test_labels)
    ]
    test_predictions = pd.DataFrame(
        {
            "text": test_texts,
            "actual": test_label_names,
            "predicted": pred_label_names,
            "pred_score": pred_scores,
        }
    )
    test_predictions.to_csv("test_predictions_bm25.csv")


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    if not config["EVALUATION"]["USE_BM25"]:
        evaluate(config)
    else:
        evaluate_bm25(config)
