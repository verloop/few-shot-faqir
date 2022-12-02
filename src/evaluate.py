import yaml

from src.evaluation.evaluate_bert_classifier import evaluate_bert_classifier
from src.evaluation.evaluate_bert_cross_encoder import evaluate_bert_crossencoder
from src.evaluation.evaluate_bert_embs import evaluate_bert_embeddings
from src.evaluation.evaluate_bm25 import evaluate_bm25
from src.evaluation.evaluate_fasttext import evaluate_fasttext
from src.evaluation.evaluate_glove import evaluate_glove
from src.evaluation.evaluate_sbert_cross_encoder import evaluate_sbert_crossencoder
from src.evaluation.evaluate_sparse_embs import evaluate_sparse_embeddings
from src.utils.utils import parse_eval_metrics


def evaluate(config):
    """
    Evaluation based on the evaluation method setup in the config.yaml file
    Saves the results in the root folder
    """

    if config["EVALUATION"]["EVALUATION_METHOD"] == "BERT_EMBEDDINGS":
        eval_metrics = evaluate_bert_embeddings(config)

    elif config["EVALUATION"]["EVALUATION_METHOD"] in [  # isort:skip
        "TFIDF_WORD_EMBEDDINGS",
        "TFIDF_CHAR_EMBEDDINGS",
        "CV_EMBEDDINGS",
    ]:
        eval_metrics = evaluate_sparse_embeddings(config)

    elif config["EVALUATION"]["EVALUATION_METHOD"] == "FASTTEXT":
        eval_metrics = evaluate_fasttext(config)

    elif config["EVALUATION"]["EVALUATION_METHOD"] == "GLOVE":
        eval_metrics = evaluate_glove(config)

    elif config["EVALUATION"]["EVALUATION_METHOD"] == "CROSS_ENCODER":
        eval_metrics = evaluate_bert_crossencoder(config)

    elif config["EVALUATION"]["EVALUATION_METHOD"] == "SBERT_CROSS_ENCODER":
        eval_metrics = evaluate_sbert_crossencoder(config)

    elif config["EVALUATION"]["EVALUATION_METHOD"] == "BERT_CLASSIFIER":
        eval_metrics = evaluate_bert_classifier(config)

    elif config["EVALUATION"]["EVALUATION_METHOD"] == "BM25":
        eval_metrics = evaluate_bm25(config)
    else:
        print("Evaluation method not supported")
        eval_metrics = {}

    if len(eval_metrics) != 0:
        evaluation_metrics = parse_eval_metrics(eval_metrics, config=config)
        evaluation_metrics.to_csv("results.csv", index=False)
    return eval_metrics


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)

    evaluate(config)
