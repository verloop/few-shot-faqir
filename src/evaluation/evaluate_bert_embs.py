import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.embeddings.dense_embeddings import (  # isort:skip
    DenseEmbeddings,
    get_similar,
)


from src.utils.utils import (  # isort:skip
    get_batched_embeddings_dense,
    get_dataloader_class,
    run_evaluation_metrics,
)


def evaluate_bert_embeddings(config):
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
    print("Evaluating bert based model embeddings")
    model_name = config["EVALUATION"]["MODEL_NAME"]
    tokenizer_model_name = config["EVALUATION"]["TOKENIZER_NAME"]
    model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    dl_train_data, _ = dl_train.get_dataloader(tokenizer=model_tokenizer)
    dl_test_data, _ = dl_test.get_dataloader(shuffle=False, tokenizer=model_tokenizer)
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
    fname = f"{config['EVALUATION']['MODEL_NAME']}_{config['DATASETS']['DATASET_NAME']}"
    fname = fname.replace("/", "")
    test_predictions.to_csv(f"predictions_{fname}.csv")

    return eval_metrics
