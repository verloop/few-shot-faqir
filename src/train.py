import yaml
from transformers import AutoTokenizer

from src.data.dataloaders import HaptikDataLoader
from src.evaluate import evaluate
from src.training.train_biencoder import BiEncoderModelTrainer
from src.training.train_crossencoder import CrossEncoderModelTrainer
from src.utils.utils import get_dataloader_class


def train(config):
    dataloader = get_dataloader_class(config)
    data_source = config["DATASETS"]["DATASET_SOURCE"]
    dataset_name = config["DATASETS"]["DATASET_NAME"]
    batch_size = config["TRAINING"]["BATCH_SIZE"]
    dl_train = dataloader(
        data_source=data_source, dataset_name=dataset_name, data_type="train"
    )

    if config["TRAINING"]["MODEL_TYPE"] == "BI_ENCODER":
        print("Training and evaluation with Bi-Encoder")
        trainer = BiEncoderModelTrainer(config["TRAINING"]["MODEL_NAME"])
        train_dataloader, _ = dl_train.get_qp_sbert_dataloader(batch_size=batch_size)
        model_folder = trainer.train(train_dataloader, config)
        config["EMBEDDINGS"]["MODEL_NAME"] = model_folder
        config["EVALUATION"]["EVALUATION_METHOD"] == "EMBEDDINGS"
        eval_metrics = evaluate(config)
        return eval_metrics

    if config["TRAINING"]["MODEL_TYPE"] == "CROSS_ENCODER":
        print("Training and evaluation with Cross Encoder")
        tokenizer = AutoTokenizer.from_pretrained(config["TRAINING"]["TOKENIZER_NAME"])
        train_dataloader, val_dataloader = dl_train.get_qp_dataloader(
            tokenizer=tokenizer, batch_size=batch_size, val_split_pct=0.2
        )
        trainer = CrossEncoderModelTrainer(config)
        model_folder = trainer.train(train_dataloader, val_dataloader)
        print(model_folder)
        config["TRAINING"]["MODEL_NAME"] = model_folder
        config["EVALUATION"]["EVALUATION_METHOD"] == "CROSS_ENCODER"
        eval_metrics = evaluate(config)
        return eval_metrics

    if config["TRAINING"]["MODEL_TYPE"] == "CLASSIFIER":
        print("Training and evaluation with classifier")
        pass


if __name__ == "__main__":
    with open("src/config/config.yaml", "r") as yamlfile:
        config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        train(config)
