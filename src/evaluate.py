import re

import numpy as np
import pandas as pd
import rank_bm25
import spacy
import yaml
from transformers import AutoTokenizer

from src.embeddings.sparse_embeddings import SparseEmbedding

from src.embeddings.dense_embeddings import (  # isort:skip
    DenseEmbeddings,
    FasttextEmbeddings,
    GlovetEmbeddings,
    get_similar,
)

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
    if config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] == False:
        if config["EMBEDDINGS"]["EMBEDDING_TYPE"] == "dense":
            print("Evaluating dense embeddings")
            model_name = config["EMBEDDINGS"]["MODEL_NAME"]
            model_tokenizer = AutoTokenizer.from_pretrained(model_name)
            dl_train_data = dl_train.get_dataloader(tokenizer=model_tokenizer)
            dl_test_data = dl_test.get_dataloader(
                shuffle=False, tokenizer=model_tokenizer
            )
            emb_model = DenseEmbeddings(model_name)
            train_embeddings, train_labels, train_texts = get_batched_embeddings_dense(
                dl_train_data, emb_model
            )
            test_embeddings, test_labels, test_texts = get_batched_embeddings_dense(
                dl_test_data, emb_model
            )
        else:
            print("Evaluating sparse embeddings")
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
    if config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] == True:
        if config["EMBEDDINGS"]["SELECT_BM25_FASTTEXT_GLOVE"] == "FASTTEXT":
            print("Evaluating Fasttext")
            dl_train_data = dl_train.get_dataloader()
            dl_test_data = dl_test.get_dataloader(shuffle=False)
            emb_model = FasttextEmbeddings(
                model_path=config["EMBEDDINGS"]["FASTTEXT_MODEL_PATH"]
            )
            train_embeddings, train_labels, train_texts = get_batched_embeddings_sparse(
                dl_train_data, emb_model
            )
            test_embeddings, test_labels, test_texts = get_batched_embeddings_sparse(
                dl_test_data, emb_model
            )
        if config["EMBEDDINGS"]["SELECT_BM25_FASTTEXT_GLOVE"] == "GLOVE":
            print("Evaluating Glove")
            dl_train_data = dl_train.get_dataloader()
            dl_test_data = dl_test.get_dataloader(shuffle=False)
            emb_model = GlovetEmbeddings(
                model_path=config["EMBEDDINGS"]["GLOVE_MODEL_PATH"]
            )
            # train_data = get_text_from_dl(dl_train_data)
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
        predicted_labels = train_labels[indx]
        pred_labels.append(predicted_labels)
        pred_scores.append(scores[0])

    oos_label_indx = None
    if config["EVALUATION"]["CHECK_OOS_ACCURACY"]:
        oos_label_indx = dl_train.dataset.intent_label_to_idx[
            config["DATASETS"]["OOS_CLASS_NAME"]
        ]
    eval_metrics = run_evaluation_metrics(
        config, test_labels_, pred_labels, pred_scores, oos_label_indx
    )

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
    if config["EMBEDDINGS"]["EMBEDDING_TYPE"] == "dense":
        fname = (
            f"{config['EMBEDDINGS']['MODEL_NAME']}_{config['DATASETS']['DATASET_NAME']}"
        )
    else:
        fname = f"{config['EMBEDDINGS']['SPARSE_EMB_METHOD']}_{config['DATASETS']['DATASET_NAME']}"
    fname = fname.replace("/", "")
    test_predictions.to_csv(f"predictions_{fname}.csv")
    return eval_metrics


def evaluate_bm25(config):
    print("Evaluating BM25")
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
                    preprocessed_text.append(tok.text)
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

    # Get train and test labels
    train_labels = []
    train_data = []
    for batch in dl_train_data:
        for label, text in zip(batch[1], batch[2]):
            train_labels.append(label)
            train_data.append(text)

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
        predicted_labels = train_labels[list(most_similar_idxs)]
        pred_labels.append(predicted_labels)
        pred_scores.append(most_similar_scores[0])
        test_labels_.append(list(label_batch))
        test_labels.append(label_batch[0])
        preds.append(most_similar_scores)

    oos_label_indx = None
    if config["EVALUATION"]["CHECK_OOS_ACCURACY"]:
        oos_label_indx = dl_train.dataset.intent_label_to_idx[
            config["DATASETS"]["OOS_CLASS_NAME"]
        ]

    eval_metrics = run_evaluation_metrics(
        config, test_labels_, pred_labels, pred_scores, oos_label_indx
    )

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
    test_predictions.to_csv("predictions_bm25.csv")
    return eval_metrics


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    if (
        not config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"]
        and config["EMBEDDINGS"]["SELECT_BM25_FASTTEXT_GLOVE"] != "BM25"
    ):
        evaluate(config)
    else:
        evaluate_bm25(config)
