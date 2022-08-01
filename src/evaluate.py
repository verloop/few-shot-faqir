import os
import re

import numpy as np
import pandas as pd
import rank_bm25
import spacy
import torch
import torch.nn.functional as F
import yaml
from sentence_transformers.cross_encoder import CrossEncoder
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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
    if config["EVALUATION"]["EVALUATION_METHOD"] == "EMBEDDINGS":
        if config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] == False:
            if config["EMBEDDINGS"]["EMBEDDING_TYPE"] == "dense":
                print("Evaluating dense embeddings")
                model_name = config["EVALUATION"]["MODEL_NAME"]
                tokenizer_model_name = config["EVALUATION"]["TOKENIZER_NAME"]
                model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
                dl_train_data, _ = dl_train.get_dataloader(tokenizer=model_tokenizer)
                dl_test_data, _ = dl_test.get_dataloader(
                    shuffle=False, tokenizer=model_tokenizer
                )
                emb_model = DenseEmbeddings(
                    model_name=model_name,
                    tokenizer_model_name=tokenizer_model_name,
                    device=device,
                )
                (
                    train_embeddings,
                    train_labels,
                    train_texts,
                ) = get_batched_embeddings_dense(dl_train_data, emb_model)
                test_embeddings, test_labels, test_texts = get_batched_embeddings_dense(
                    dl_test_data, emb_model
                )
            else:
                print("Evaluating sparse embeddings")
                dl_train_data, _ = dl_train.get_dataloader()
                dl_test_data, _ = dl_test.get_dataloader(shuffle=False)
                emb_model = SparseEmbedding(
                    sparse_embedding_method=config["EMBEDDINGS"]["SPARSE_EMB_METHOD"]
                )
                train_data = get_text_from_dl(dl_train_data)
                emb_model.train(train_data)
                (
                    train_embeddings,
                    train_labels,
                    train_texts,
                ) = get_batched_embeddings_sparse(dl_train_data, emb_model)
                (
                    test_embeddings,
                    test_labels,
                    test_texts,
                ) = get_batched_embeddings_sparse(dl_test_data, emb_model)

        if config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] == True:
            if config["EMBEDDINGS"]["SELECT_BM25_FASTTEXT_GLOVE"] == "FASTTEXT":
                print("Evaluating Fasttext")
                dl_train_data, _ = dl_train.get_dataloader()
                dl_test_data, _ = dl_test.get_dataloader(shuffle=False)
                emb_model = FasttextEmbeddings(
                    model_path=config["EMBEDDINGS"]["FASTTEXT_MODEL_PATH"]
                )
                (
                    train_embeddings,
                    train_labels,
                    train_texts,
                ) = get_batched_embeddings_sparse(dl_train_data, emb_model)
                (
                    test_embeddings,
                    test_labels,
                    test_texts,
                ) = get_batched_embeddings_sparse(dl_test_data, emb_model)
            if config["EMBEDDINGS"]["SELECT_BM25_FASTTEXT_GLOVE"] == "GLOVE":
                print("Evaluating Glove")
                dl_train_data, _ = dl_train.get_dataloader()
                dl_test_data, _ = dl_test.get_dataloader(shuffle=False)
                emb_model = GlovetEmbeddings(
                    model_path=config["EMBEDDINGS"]["GLOVE_MODEL_PATH"]
                )
                (
                    train_embeddings,
                    train_labels,
                    train_texts,
                ) = get_batched_embeddings_sparse(dl_train_data, emb_model)
                (
                    test_embeddings,
                    test_labels,
                    test_texts,
                ) = get_batched_embeddings_sparse(dl_test_data, emb_model)

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
        if config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] == True:
            fname = f"{config['EMBEDDINGS']['SELECT_BM25_FASTTEXT_GLOVE']}"
        else:
            if config["EMBEDDINGS"]["EMBEDDING_TYPE"] == "dense":
                fname = f"{config['EVALUATION']['MODEL_NAME']}_{config['DATASETS']['DATASET_NAME']}"
            else:
                fname = f"{config['EMBEDDINGS']['SPARSE_EMB_METHOD']}_{config['DATASETS']['DATASET_NAME']}"
        fname = fname.replace("/", "")
        test_predictions.to_csv(f"predictions_{fname}.csv")
        return eval_metrics

    if config["EVALUATION"]["EVALUATION_METHOD"] == "CROSS_ENCODER":
        print("Running evaluation with cross encoders")
        batch_size = config["EVALUATION"]["BATCH_SIZE"]
        tokenizer = AutoTokenizer.from_pretrained(
            config["EVALUATION"]["TOKENIZER_NAME"]
        )
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
        # test_predictions = pd.DataFrame(
        #     {
        #         "text": test_texts,
        #         "actual": test_label_names,
        #         "predicted": pred_label_names,
        #         "pred_score": pred_scores,
        #     }
        # )
        # test_predictions.to_csv(f"predictions_cross_encoder.csv")
        return eval_metrics

    if config["EVALUATION"]["EVALUATION_METHOD"] == "SBERT_CROSS_ENCODER":
        print("Running evaluation with sbert cross encoders")
        batch_size = config["EVALUATION"]["BATCH_SIZE"]
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
            batch_size=batch_size
        )
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
        test_predictions.to_csv(f"predictions_cross_encoder.csv")
        return eval_metrics

    if config["EVALUATION"]["EVALUATION_METHOD"] == "CLASSIFIER":
        print("Running evaluation with classifier approach")
        num_labels = config["DATASETS"]["N_LABELS"]
        model_name = config["EVALUATION"]["MODEL_NAME"]
        batch_size = config["EVALUATION"]["BATCH_SIZE"]
        tokenizer = AutoTokenizer.from_pretrained(
            config["EVALUATION"]["TOKENIZER_NAME"]
        )
        test_dataloader, _ = dl_test.get_dataloader(
            batch_size=batch_size, tokenizer=tokenizer
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        model.to(device)
        model.eval()
        prediction_labels_test = []
        true_labels_test = []
        pred_probs = []
        for batch in test_dataloader:
            true_labels_test += batch[1].numpy().flatten().tolist()
            with torch.no_grad():
                batch[0].to(torch.device(device))
                outputs = model(**batch[0])
                prediction_probs = F.softmax(outputs.logits, dim=1)
                predicted = torch.argsort(outputs.logits, 1, descending=True)
                predicted = predicted.detach().cpu().numpy()
                prediction_probs = prediction_probs.detach().cpu().numpy()
                pred_probs_ = [
                    list(i[j])[0] for i, j in zip(prediction_probs, predicted)
                ]
                pred_probs += pred_probs_
                prediction_labels_test += list(predicted)

        true_labels_test = [[i] for i in true_labels_test]

        oos_label_indx = None
        if config["EVALUATION"]["CHECK_OOS_ACCURACY"]:
            oos_label_indx = dl_train.dataset.intent_label_to_idx[
                config["DATASETS"]["OOS_CLASS_NAME"]
            ]

        eval_metrics = run_evaluation_metrics(
            config, true_labels_test, prediction_labels_test, pred_probs, oos_label_indx
        )
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
    dl_train_data, _ = dl_train.get_dataloader(batch_size=100, shuffle=False)
    dl_test_data, _ = dl_test.get_dataloader(batch_size=1, shuffle=False)

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
    if not config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"] or (
        config["EMBEDDINGS"]["USE_BM25_FASTTEXT_GLOVE"]
        and config["EMBEDDINGS"]["SELECT_BM25_FASTTEXT_GLOVE"] != "BM25"
    ):
        evaluate(config)
    else:
        evaluate_bm25(config)
