import pandas as pd
import yaml
from transformers import AutoTokenizer

from src.embeddings.dense_embeddings import DenseEmbeddings, get_similar
from src.embeddings.sparse_embeddings import SparseEmbedding

from src.utils.utils import (  # isort:skip
    get_batched_embeddings_dense,
    get_batched_embeddings_sparse,
    get_dataloader,
    get_train_text,
    run_evaluation_metrics,
)

with open("src/config/config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)


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
    train_data = get_train_text(dl_train_data)
    emb_model.train(train_data)
    train_embeddings, train_labels, train_texts = get_batched_embeddings_sparse(
        dl_train_data, emb_model
    )
    test_embeddings, test_labels, test_texts = get_batched_embeddings_sparse(
        dl_test_data, emb_model
    )


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
